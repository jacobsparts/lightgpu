//! A read-only memory mapping of a file.
//!
//! Every engine here loads its weights by `mmap`ing the checkpoint and reading
//! tensors out of the mapping, so that a 200 MB blob is never copied into the
//! heap and a tensor can go straight to VRAM with one `cudaMemcpy`. This is that
//! mapping, factored out once.
//!
//! The mapping is `PROT_READ` + `MAP_PRIVATE`, so writes to it are impossible
//! and the file is never modified. `madvise(MADV_SEQUENTIAL)` asks the kernel to
//! read ahead, which turns a loader that walks the tensors in file order into
//! one streaming read.
//!
//! [`File::open`] maps the file and [`File::read`] reads it into the heap
//! instead; an engine that keeps no file handle open - or that fills a device
//! buffer in the background - can use either, and both give the same
//! [`File::as_slice`].

use std::ffi::c_void;
use std::fs::File as StdFile;
use std::io::Read;
use std::os::unix::io::AsRawFd;
use std::path::Path;

pub struct File {
    /// Non-null for a mapping, null for the heap-backed variant.
    ptr: *const u8,
    len: usize,
    owned: Option<Vec<u8>>,
    path: std::path::PathBuf,
}

// SAFETY: the region is read-only and immutable after construction; the raw
// pointer is never written through.
unsafe impl Send for File {}
unsafe impl Sync for File {}

impl File {
    /// Map `path` read-only. The file is not modified and the mapping is never
    /// written through.
    pub fn open(path: impl AsRef<Path>) -> Result<File, String> {
        let path = path.as_ref();
        let f = StdFile::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let len = f.metadata().map_err(|e| e.to_string())?.len() as usize;
        if len == 0 {
            return Err(format!("{}: file is empty", path.display()));
        }
        // SAFETY: `f` is open for the call; PROT_READ means the mapping cannot
        // fault by writing, and MAP_PRIVATE keeps the modification out of the
        // file. The mapping lives until `Drop`.
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                f.as_raw_fd(),
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return Err(format!(
                "{}: mmap: {}",
                path.display(),
                std::io::Error::last_os_error()
            ));
        }
        // SAFETY: `p` is a valid mapping of `len` bytes for the call.
        unsafe { libc::madvise(p, len, libc::MADV_SEQUENTIAL) };
        Ok(File { ptr: p as *const u8, len, owned: None, path: path.to_path_buf() })
    }

    /// Read `path` into the heap rather than mapping it. A device buffer can
    /// then be filled from stable memory with no page faults mid-transfer.
    pub fn read(path: impl AsRef<Path>) -> Result<File, String> {
        let path = path.as_ref();
        let mut f = StdFile::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).map_err(|e| format!("{}: {e}", path.display()))?;
        let ptr = buf.as_ptr();
        Ok(File { ptr, len: buf.len(), owned: Some(buf), path: path.to_path_buf() })
    }

    /// The whole file as one slice. Valid for the lifetime of `self`.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr`/`len` describe a readable region for `self`'s lifetime:
        // either a live mapping or the heap buffer in `owned`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The path the file was opened from, for error messages.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for File {
    fn drop(&mut self) {
        if self.owned.is_none() && !self.ptr.is_null() {
            // SAFETY: `ptr`/`len` came from `mmap` in `open` and have not been
            // unmapped since.
            unsafe { libc::munmap(self.ptr as *mut c_void, self.len) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("lightgpu-mmap-{}-{}", std::process::id(), name));
        p
    }

    #[test]
    fn maps_and_reads_the_same_bytes() {
        let p = tmp("bytes");
        std::fs::write(&p, b"hello mapping").unwrap();
        let mapped = File::open(&p).unwrap();
        let heap = File::read(&p).unwrap();
        assert_eq!(mapped.as_slice(), b"hello mapping");
        assert_eq!(heap.as_slice(), mapped.as_slice());
        assert_eq!(mapped.len(), 13);
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn refuses_an_empty_file() {
        let p = tmp("empty");
        std::fs::write(&p, b"").unwrap();
        assert!(File::open(&p).is_err());
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn reports_the_source_path() {
        let p = tmp("path");
        std::fs::write(&p, b"x").unwrap();
        let f = File::open(&p).unwrap();
        assert_eq!(f.path(), p.as_path());
        std::fs::remove_file(&p).unwrap();
    }
}
