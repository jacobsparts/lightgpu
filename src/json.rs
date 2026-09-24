//! A minimal JSON reader - enough for a safetensors header.
//!
//! A safetensors header is JSON, and this toolkit's whole point is to stay
//! dependency-light, so rather than pull in serde this is a small,
//! strict-enough recursive-descent parser for exactly the subset such a header
//! uses: objects, arrays, strings, numbers and the booleans/null.
//! Floating-point literals are parsed as `Num`, and [`Value::as_usize`] extracts
//! an exact non-negative integer from one.
//!
//! It is deliberately not a general JSON library: no escapes beyond the
//! standard ones, no unicode surrogate pairs, and duplicate keys collapse to the
//! last occurrence. Tensor names and metadata values in these checkpoints are
//! ASCII, and anything outside the subset is a hard error rather than a silent
//! misread.

use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Value>),
    Obj(BTreeMap<String, Value>),
}

impl Value {
    pub fn as_object(&self) -> Option<&BTreeMap<String, Value>> {
        match self {
            Value::Obj(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Vec<Value>> {
        match self {
            Value::Arr(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Exact integer extraction. Rejects a negative or fractional value, which
    /// a shape or an offset must never be.
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Value::Num(n) if *n >= 0.0 && n.fract() == 0.0 => Some(*n as usize),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.as_object().and_then(|m| m.get(key))
    }
}

pub fn parse(s: &[u8]) -> Result<Value, String> {
    let mut p = Parser { s, i: 0 };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.s.len() {
        return Err(format!("trailing bytes after JSON value at {}", p.i));
    }
    Ok(v)
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn peek(&self) -> Result<u8, String> {
        self.s.get(self.i).copied().ok_or_else(|| "unexpected end of JSON".into())
    }

    fn value(&mut self) -> Result<Value, String> {
        match self.peek()? {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => Ok(Value::Str(self.string()?)),
            b't' | b'f' | b'n' => self.literal(),
            _ => self.number(),
        }
    }

    fn literal(&mut self) -> Result<Value, String> {
        for (word, val) in [("true", Value::Bool(true)), ("false", Value::Bool(false)), ("null", Value::Null)] {
            if self.s[self.i..].starts_with(word.as_bytes()) {
                self.i += word.len();
                // `null`/`true`/`false` must not run into a longer word.
                if self.i < self.s.len()
                    && (self.s[self.i].is_ascii_alphanumeric() || self.s[self.i] == b'_')
                {
                    continue;
                }
                return Ok(val);
            }
        }
        Err(format!("bad literal at {}", self.i))
    }

    fn number(&mut self) -> Result<Value, String> {
        let start = self.i;
        if self.i < self.s.len() && (self.s[self.i] == b'-' || self.s[self.i] == b'+') {
            self.i += 1;
        }
        while self.i < self.s.len()
            && (self.s[self.i].is_ascii_digit() || matches!(self.s[self.i], b'.' | b'e' | b'E' | b'-' | b'+'))
        {
            self.i += 1;
        }
        let text = std::str::from_utf8(&self.s[start..self.i]).map_err(|e| e.to_string())?;
        text.parse::<f64>()
            .map(Value::Num)
            .map_err(|e| format!("bad number `{text}` at {start}: {e}"))
    }

    fn string(&mut self) -> Result<String, String> {
        if self.peek()? != b'"' {
            return Err(format!("expected string at {}", self.i));
        }
        self.i += 1;
        let mut out = String::new();
        loop {
            let c = self.peek()?;
            self.i += 1;
            match c {
                b'"' => return Ok(out),
                b'\\' => {
                    let e = self.peek()?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hex = self
                                .s
                                .get(self.i..self.i + 4)
                                .ok_or("truncated \\u escape")?;
                            self.i += 4;
                            let code = u32::from_str_radix(
                                std::str::from_utf8(hex).map_err(|e| e.to_string())?,
                                16,
                            )
                            .map_err(|e| format!("bad \\u escape: {e}"))?;
                            out.push(char::from_u32(code).ok_or("bad code point")?);
                        }
                        other => return Err(format!("bad escape \\{}", other as char)),
                    }
                }
                _ => {
                    // Copy the whole UTF-8 run up to the next quote/backslash.
                    let start = self.i - 1;
                    while self.i < self.s.len() && self.s[self.i] != b'"' && self.s[self.i] != b'\\' {
                        self.i += 1;
                    }
                    out.push_str(
                        std::str::from_utf8(&self.s[start..self.i]).map_err(|e| e.to_string())?,
                    );
                }
            }
        }
    }

    fn array(&mut self) -> Result<Value, String> {
        self.i += 1; // [
        let mut out = Vec::new();
        self.ws();
        if self.peek()? == b']' {
            self.i += 1;
            return Ok(Value::Arr(out));
        }
        loop {
            self.ws();
            out.push(self.value()?);
            self.ws();
            match self.peek()? {
                b',' => self.i += 1,
                b']' => {
                    self.i += 1;
                    return Ok(Value::Arr(out));
                }
                _ => return Err(format!("expected , or ] at {}", self.i)),
            }
        }
    }

    fn object(&mut self) -> Result<Value, String> {
        self.i += 1; // {
        let mut out = BTreeMap::new();
        self.ws();
        if self.peek()? == b'}' {
            self.i += 1;
            return Ok(Value::Obj(out));
        }
        loop {
            self.ws();
            let k = self.string()?;
            self.ws();
            if self.peek()? != b':' {
                return Err(format!("expected : at {}", self.i));
            }
            self.i += 1;
            self.ws();
            let v = self.value()?;
            out.insert(k, v);
            self.ws();
            match self.peek()? {
                b',' => self.i += 1,
                b'}' => {
                    self.i += 1;
                    return Ok(Value::Obj(out));
                }
                _ => return Err(format!("expected , or }} at {}", self.i)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_safetensors_header() {
        let h = br#"{"__metadata__":{"scale":"4"},"conv_first.weight":{"dtype":"F32","shape":[64,3,3,3],"data_offsets":[0,6912]}}"#;
        let v = parse(h).unwrap();
        let m = v.get("__metadata__").unwrap();
        assert_eq!(m.get("scale").unwrap().as_str(), Some("4"));
        let t = v.get("conv_first.weight").unwrap();
        assert_eq!(t.get("shape").unwrap().as_array().unwrap().len(), 4);
        assert_eq!(t.get("data_offsets").unwrap().as_array().unwrap()[1].as_usize(), Some(6912));
        assert_eq!(t.get("dtype").unwrap().as_str(), Some("F32"));
    }

    #[test]
    fn rejects_trailing_junk() {
        assert!(parse(b"{}x").is_err());
    }

    #[test]
    fn handles_escapes() {
        let v = parse(br#"{"a\u002fb":"x\ny"}"#).unwrap();
        assert_eq!(v.get("a/b").unwrap().as_str(), Some("x\ny"));
    }
}
