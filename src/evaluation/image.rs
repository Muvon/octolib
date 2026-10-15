// Copyright 2026 Muvon Un Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Jev image limits: <https://docs.typesafe.ai/go/images>, October 2026 (inclusive).

use super::types::EvaluationImageFormat;

pub(crate) const MAX_FILE_BYTES: usize = 16 * 1024 * 1024;
pub(crate) const MIN_SIDE: u32 = 256;
pub(crate) const MAX_SIDE: u32 = 1568;
pub(crate) const MIN_PIXELS: u64 = 78_400;
pub(crate) const MAX_PIXELS: u64 = 2_458_624;

pub(crate) fn detect(data: &[u8]) -> Result<(EvaluationImageFormat, (u32, u32)), &'static str> {
    if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        return png_dimensions(data).map(|dimensions| (EvaluationImageFormat::Png, dimensions));
    }
    if data.starts_with(b"\xff\xd8") {
        return jpeg_dimensions(data).map(|dimensions| (EvaluationImageFormat::Jpeg, dimensions));
    }
    Err("unsupported image format; only PNG or JPEG still images are allowed")
}

fn png_dimensions(data: &[u8]) -> Result<(u32, u32), &'static str> {
    let invalid = || "truncated or corrupt PNG chunk header";
    let mut chunks = data.get(8..).ok_or_else(invalid)?;
    if read::<8>(&mut chunks) != Some(*b"\0\0\0\rIHDR") {
        return Err(invalid());
    }
    let width = u32::from_be_bytes(read(&mut chunks).ok_or_else(invalid)?);
    let height = u32::from_be_bytes(read(&mut chunks).ok_or_else(invalid)?);
    read::<5>(&mut chunks).ok_or_else(invalid)?;
    read::<4>(&mut chunks).ok_or_else(invalid)?;
    loop {
        let length = u32::from_be_bytes(read(&mut chunks).ok_or_else(invalid)?);
        let kind = read::<4>(&mut chunks).ok_or_else(invalid)?;
        let length = usize::try_from(length).map_err(|_| invalid())?;
        let (_, rest) = chunks.split_at_checked(length).ok_or_else(invalid)?;
        chunks = rest;
        read::<4>(&mut chunks).ok_or_else(invalid)?;
        match &kind {
            b"acTL" => {
                return Err("animated PNG (acTL) is not allowed; only still images are supported");
            }
            b"IDAT" => return Ok((width, height)),
            b"IHDR" | b"IEND" => return Err(invalid()),
            _ => {}
        }
    }
}

fn jpeg_dimensions(data: &[u8]) -> Result<(u32, u32), &'static str> {
    let invalid = || "truncated or corrupt JPEG header; a valid SOF before SOS/end is required";
    let mut segments = data.get(2..).ok_or_else(invalid)?;
    loop {
        if read::<1>(&mut segments) != Some([0xff]) {
            return Err(invalid());
        }
        let [mut marker] = read(&mut segments).ok_or_else(invalid)?;
        while marker == 0xff {
            let [next] = read(&mut segments).ok_or_else(invalid)?;
            marker = next;
        }
        match marker {
            0x00 | 0xd8 | 0xd9 | 0xda => return Err(invalid()),
            0xd0..=0xd7 | 0x01 => continue,
            _ => {}
        }
        let length = u16::from_be_bytes(read(&mut segments).ok_or_else(invalid)?);
        let length = usize::from(length.checked_sub(2).ok_or_else(invalid)?);
        let (mut payload, rest) = segments.split_at_checked(length).ok_or_else(invalid)?;
        segments = rest;
        if matches!(marker, 0xc0..=0xcf) && !matches!(marker, 0xc4 | 0xc8 | 0xcc) {
            let [_, h0, h1, w0, w1, components] = read(&mut payload).ok_or_else(invalid)?;
            if components == 0 || length != 6 + 3 * usize::from(components) {
                return Err(invalid());
            }
            return Ok((
                u32::from(u16::from_be_bytes([w0, w1])),
                u32::from(u16::from_be_bytes([h0, h1])),
            ));
        }
    }
}

fn read<const N: usize>(data: &mut &[u8]) -> Option<[u8; N]> {
    let (bytes, rest) = data.split_first_chunk::<N>()?;
    *data = rest;
    Some(*bytes)
}

#[cfg(test)]
#[path = "image_tests.rs"]
pub(crate) mod tests;
