//! A reader for the standard `.safetensors` container.
//!
//! The container is
//!
//! ```text
//! u64 header_len (little endian) | header_len bytes of JSON | tensor payloads
//! ```
//!
//! where the JSON header holds one entry per tensor - `dtype`, `shape` and
//! `data_offsets` - plus an optional `__metadata__` of string pairs. This module
//! maps the file, parses that header, and hands out borrowed slices into the
//! mapping: a tensor can go straight to VRAM with one `cudaMemcpy`, and the
//! container itself costs nothing beyond the JSON index.
//!
//! Offsets are reported **absolute in the file**, not relative to the payload
//! section as the container encodes them, so a caller that copies the whole
//! mapping into a device buffer and then adds `info.offset` gets a correct
//! device pointer. That is how an engine with a single weight arena uses this.
//!
//! What this is not: a general safetensors implementation. It is deliberately
//! small and dependency-light (no serde, no `safetensors` crate) and it is
//! strict rather than permissive about anything that would let a malformed file
//! reach an unchecked slice:
//!
//! * the header must parse as a JSON object and `8 + header_len` must be inside
//!   the file;
//! * every tensor must name a known dtype, carry exactly two offsets with
//!   `end >= start`, and lie inside the mapping;
//! * a shape entry must be a non-negative integer (a wrong or fractional entry
//!   is an error, not a silently dropped dimension);
//! * `nbytes` must equal `numel * dtype_size`, computed with checked arithmetic.
//!
//! It does not check the container's ordering/contiguity rules, and it does not
//! read anything but the index - the payloads are the file's own bytes.

use std::collections::BTreeMap;
use std::path::Path;

use crate::json::Value;
use crate::mmap;

/// Element type of a tensor payload.
///
/// The two engines sharing this reader use `F32` only; the rest are recognised
/// so that an exotic file is rejected with a clear message rather than read as
/// if it were FP32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Bool,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F16,
    BF16,
    F32,
    F64,
}

impl DType {
    pub fn parse(s: &str) -> Result<DType, String> {
        Ok(match s {
            "BOOL" => DType::Bool,
            "U8" => DType::U8,
            "I8" => DType::I8,
            "U16" => DType::U16,
            "I16" => DType::I16,
            "U32" => DType::U32,
            "I32" => DType::I32,
            "U64" => DType::U64,
            "I64" => DType::I64,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            "F32" => DType::F32,
            "F64" => DType::F64,
            other => return Err(format!("unknown safetensors dtype `{other}`")),
        })
    }

    /// Size of one element in bytes.
    pub fn size(&self) -> usize {
        match self {
            DType::Bool | DType::U8 | DType::I8 => 1,
            DType::U16 | DType::I16 | DType::F16 | DType::BF16 => 2,
            DType::U32 | DType::I32 | DType::F32 => 4,
            DType::U64 | DType::I64 | DType::F64 => 8,
        }
    }
}

/// One tensor's entry in the header, with `offset` made absolute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    /// Byte offset of the payload in the file (absolute, not relative to the
    /// payload section).
    pub offset: usize,
    /// Length of the payload in bytes: `numel * dtype.size()`, already checked.
    pub nbytes: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A mapped `.safetensors` file and its tensor index.
pub struct File {
    map: mmap::File,
    tensors: BTreeMap<String, TensorInfo>,
    order: Vec<String>,
    metadata: BTreeMap<String, String>,
    /// File offset of the first payload byte (the data section).
    data_start: usize,
}

impl File {
    pub fn open(path: impl AsRef<Path>) -> Result<File, String> {
        let map = mmap::File::open(path.as_ref())?;
        File::from_mapping(map)
    }

    /// Read the file into the heap instead of mapping it. Same accessors; use
    /// this if something perturbs the file while the process runs, or if a
    /// device copy should not fault pages in as it goes.
    pub fn read(path: impl AsRef<Path>) -> Result<File, String> {
        let map = mmap::File::read(path.as_ref())?;
        File::from_mapping(map)
    }

    fn from_mapping(map: mmap::File) -> Result<File, String> {
        let all = map.as_slice();
        if all.len() < 8 {
            return Err(format!("{}: too small to be safetensors", map.path().display()));
        }
        let hlen = u64::from_le_bytes(all[..8].try_into().unwrap()) as usize;
        let header_end = 8usize
            .checked_add(hlen)
            .ok_or_else(|| "safetensors header length overflows".to_string())?;
        if header_end > all.len() {
            return Err(format!(
                "{}: header length {hlen} exceeds file size {}",
                map.path().display(),
                all.len()
            ));
        }
        let header_bytes = &all[8..header_end];
        let root = crate::json::parse(header_bytes).map_err(|e| {
            format!("{}: safetensors header: {e}", map.path().display())
        })?;
        let obj = root.as_object().ok_or("safetensors header is not a JSON object")?;

        let mut tensors = BTreeMap::new();
        let mut order = Vec::new();
        let mut metadata = BTreeMap::new();

        for (key, entry) in obj {
            if key == "__metadata__" {
                if let Some(m) = entry.as_object() {
                    for (mk, mv) in m {
                        if let Some(s) = mv.as_str() {
                            metadata.insert(mk.clone(), s.to_string());
                        }
                    }
                }
                continue;
            }
            let t = parse_tensor(key, entry, header_end, all.len())?;
            order.push(t.name.clone());
            tensors.insert(t.name.clone(), t);
        }
        if tensors.is_empty() {
            return Err(format!("{}: header declares no tensors", map.path().display()));
        }
        Ok(File { map, tensors, order, metadata, data_start: header_end })
    }

    /// The tensor index, in the order the header lists them.
    pub fn order(&self) -> &[String] {
        &self.order
    }

    /// Metadata from the header's `__metadata__` object, if the file has one.
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    pub fn metadata_get(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }

    /// Metadata value parsed as a `usize`, with the key named on failure - the
    /// value is a string in the container, so this is the usual way to read an
    /// architecture constant out of it.
    pub fn metadata_usize(&self, key: &str) -> Result<usize, String> {
        let s = self
            .metadata
            .get(key)
            .ok_or_else(|| format!("safetensors metadata has no `{key}`"))?;
        s.parse::<usize>()
            .map_err(|e| format!("safetensors metadata `{key}` = {s:?}: {e}"))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// One tensor's header entry, with the payload's absolute file offset.
    pub fn info(&self, name: &str) -> Result<&TensorInfo, String> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("no tensor `{name}` in the checkpoint"))
    }

    /// The tensor's payload, borrowed from the mapping.
    pub fn raw(&self, name: &str) -> Result<&[u8], String> {
        let t = self.info(name)?;
        Ok(&self.map.as_slice()[t.offset..t.offset + t.nbytes])
    }

    /// The tensor's payload as `f32`, borrowed from the mapping. Rejects a
    /// tensor that is not `F32` rather than reinterpreting its bytes.
    pub fn f32(&self, name: &str) -> Result<&[f32], String> {
        let t = self.info(name)?;
        if t.dtype != DType::F32 {
            return Err(format!("tensor `{name}` is {:?}, not F32", t.dtype));
        }
        if t.offset % 4 != 0 {
            return Err(format!(
                "tensor `{name}` starts at byte {} in the file, which is not 4-byte aligned",
                t.offset
            ));
        }
        let bytes = self.raw(name)?;
        // SAFETY: the payload is `numel * 4` bytes of little-endian FP32 (checked
        // at load), the offset is 4-byte aligned and inside the mapping, and the
        // mapping is read-only and outlives the borrow.
        Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, t.numel()) })
    }

    /// The tensor's payload copied out as host `f32` - for a caller that needs
    /// to own or repack it.
    pub fn to_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        Ok(self.f32(name)?.to_vec())
    }

    /// Every tensor's shape, for a caller that validates the architecture.
    pub fn shape(&self, name: &str) -> Result<&[usize], String> {
        Ok(self.info(name)?.shape.as_slice())
    }

    /// Sum of all payload lengths. For a container with no holes this is the
    /// data section's size.
    pub fn payload_bytes(&self) -> usize {
        self.tensors.values().map(|t| t.nbytes).sum()
    }

    /// The whole file's bytes - the header too, which is what an engine that
    /// uploads the mapping in one `cudaMemcpy` wants, because every
    /// [`TensorInfo::offset`] is absolute in this slice.
    pub fn as_slice(&self) -> &[u8] {
        self.map.as_slice()
    }

    /// File offset where the payload section starts.
    pub fn data_start(&self) -> usize {
        self.data_start
    }

    /// The payload section alone, i.e. [`File::as_slice`] without the header.
    pub fn data(&self) -> &[u8] {
        &self.map.as_slice()[self.data_start..]
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn path(&self) -> &Path {
        self.map.path()
    }
}

impl std::fmt::Debug for File {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("safetensors::File")
            .field("path", &self.path())
            .field("len", &self.len())
            .field("data_start", &self.data_start)
            .field("tensors", &self.tensors.len())
            .finish()
    }
}

fn parse_tensor(
    name: &str,
    entry: &Value,
    header_end: usize,
    file_len: usize,
) -> Result<TensorInfo, String> {
    let dtype = DType::parse(
        entry
            .get("dtype")
            .and_then(|d| d.as_str())
            .ok_or_else(|| format!("tensor `{name}` has no dtype"))?,
    )
    .map_err(|e| format!("tensor `{name}`: {e}"))?;

    let shape = entry
        .get("shape")
        .and_then(|s| s.as_array())
        .ok_or_else(|| format!("tensor `{name}` has no shape"))?
        .iter()
        .map(|v| {
            v.as_usize()
                .ok_or_else(|| format!("tensor `{name}`: shape entries must be non-negative integers"))
        })
        .collect::<Result<Vec<usize>, String>>()?;

    let offsets = entry
        .get("data_offsets")
        .and_then(|s| s.as_array())
        .ok_or_else(|| format!("tensor `{name}` has no data_offsets"))?;
    if offsets.len() != 2 {
        return Err(format!(
            "tensor `{name}`: data_offsets has {} entries, expected 2",
            offsets.len()
        ));
    }
    let lo = offsets[0]
        .as_usize()
        .ok_or_else(|| format!("tensor `{name}`: data_offsets[0] is not a non-negative integer"))?;
    let hi = offsets[1]
        .as_usize()
        .ok_or_else(|| format!("tensor `{name}`: data_offsets[1] is not a non-negative integer"))?;
    if hi < lo {
        return Err(format!("tensor `{name}`: data_offsets [{lo}, {hi}] are reversed"));
    }
    let nbytes = hi - lo;

    let numel = shape.iter().try_fold(1usize, |acc, &d| {
        acc.checked_mul(d)
            .ok_or_else(|| format!("tensor `{name}`: shape overflows"))
    })?;
    let expect = numel
        .checked_mul(dtype.size())
        .ok_or_else(|| format!("tensor `{name}`: shape overflows"))?;
    if expect != nbytes {
        return Err(format!(
            "tensor `{name}`: data_offsets span {nbytes} bytes but shape {shape:?} of {:?} needs {expect}",
            dtype
        ));
    }

    let offset = header_end
        .checked_add(lo)
        .ok_or_else(|| format!("tensor `{name}`: offset overflows"))?;
    let end = offset
        .checked_add(nbytes)
        .ok_or_else(|| format!("tensor `{name}`: offset overflows"))?;
    if end > file_len {
        return Err(format!(
            "tensor `{name}`: payload at {offset}..{end} is outside the {file_len}-byte file"
        ));
    }

    Ok(TensorInfo { name: name.to_string(), dtype, shape, offset, nbytes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Write a container from a header string and raw payloads.
    fn write_container(name: &str, header: &str, payload: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("lightgpu-st-{}-{}", std::process::id(), name));
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header.as_bytes()).unwrap();
        f.write_all(payload).unwrap();
        p
    }

    #[test]
    fn reads_tensors_and_metadata() {
        // Pad the header to a multiple of 4 so the first payload is 4-byte
        // aligned, the way a real writer arranges it.
        let header = r#"{"__metadata__":{"scale":"4"},"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"b":{"dtype":"F32","shape":[3],"data_offsets":[16,28]}}   "#;
        assert_eq!((8 + header.len()) % 4, 0, "test header must be 4-byte aligned");
        let payload: Vec<u8> = (0..28u8).collect();
        let p = write_container("ok", header, &payload);
        let st = File::open(&p).unwrap();

        assert_eq!(st.metadata_get("scale"), Some("4"));
        assert_eq!(st.metadata_usize("scale").unwrap(), 4);
        assert_eq!(st.order(), &["a".to_string(), "b".to_string()]);
        assert_eq!(st.info("a").unwrap().shape, vec![2, 2]);
        assert_eq!(st.info("a").unwrap().numel(), 4);
        // Absolute, not relative to the data section.
        assert_eq!(st.info("a").unwrap().offset, st.data_start());
        assert_eq!(st.len(), 8 + header.len() + 28);
        assert_eq!(st.data(), &payload[..]);
        assert_eq!(st.raw("b").unwrap(), &payload[16..28]);
        assert_eq!(st.f32("a").unwrap().len(), 4);
        assert!(st.contains("a") && !st.contains("z"));
        assert!(st.info("z").is_err());
        assert_eq!(st.payload_bytes(), 28);
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn reads_through_the_heap_variant_too() {
        let header = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}      "#;
        assert_eq!((8 + header.len()) % 4, 0, "test header must be 4-byte aligned");
        let p = write_container("heap", header, &[1, 2, 3, 4]);
        let st = File::read(&p).unwrap();
        assert_eq!(st.f32("a").unwrap()[0].to_bits(), f32::from_le_bytes([1, 2, 3, 4]).to_bits());
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn rejects_malformed_headers() {
        let cases: &[(&str, &str)] = &[
            ("no-dtype", r#"{"a":{"shape":[1],"data_offsets":[0,4]}}"#),
            ("bad-dtype", r#"{"a":{"dtype":"Q4","shape":[1],"data_offsets":[0,4]}}"#),
            ("no-shape", r#"{"a":{"dtype":"F32","data_offsets":[0,4]}}"#),
            (
                "fractional-shape",
                r#"{"a":{"dtype":"F32","shape":[1.5],"data_offsets":[0,4]}}"#,
            ),
            (
                "negative-shape",
                r#"{"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}"#,
            ),
            (
                "one-offset",
                r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0]}}"#,
            ),
            (
                "reversed-offsets",
                r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}}"#,
            ),
            (
                "size-mismatch",
                r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
            ),
        ];
        for (name, header) in cases {
            let p = write_container(&format!("bad-{name}"), header, &[0u8; 4]);
            let err = File::open(&p).expect_err(name);
            assert!(!err.is_empty(), "{name}: expected an error");
            std::fs::remove_file(&p).unwrap();
        }
    }

    #[test]
    fn rejects_a_payload_past_the_end_of_the_file() {
        let header = r#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
        let p = write_container("short", header, &[0u8; 8]);
        let err = File::open(&p).unwrap_err();
        assert!(err.contains("outside"), "{err}");
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn rejects_a_misaligned_f32_payload() {
        // Header length 8 + 62 leaves the payload at an offset that is not a
        // multiple of 4, so `f32` must refuse it rather than build a misaligned
        // slice.
        let header = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        assert_ne!((8 + header.len()) % 4, 0, "test header must be misaligned");
        let p = write_container("misaligned", header, &[0u8; 4]);
        let st = File::open(&p).unwrap();
        let err = st.f32("a").unwrap_err();
        assert!(err.contains("aligned"), "{err}");
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn rejects_a_header_that_is_not_an_object() {
        let p = write_container("arr", "[1,2,3]", &[0u8; 4]);
        assert!(File::open(&p).is_err());
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn rejects_a_truncated_file() {
        let mut p = std::env::temp_dir();
        p.push(format!("lightgpu-st-{}-short-file", std::process::id()));
        std::fs::write(&p, [0u8; 4]).unwrap();
        assert!(File::open(&p).is_err());
        std::fs::remove_file(&p).unwrap();
    }
}
