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

use super::*;

pub(crate) fn png(width: u32, height: u32) -> Vec<u8> {
    let mut data = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut header = width.to_be_bytes().to_vec();
    header.extend_from_slice(&height.to_be_bytes());
    header.extend_from_slice(&[8, 2, 0, 0, 0]);
    png_chunk(&mut data, b"IHDR", &header);
    png_chunk(&mut data, b"IDAT", &[]);
    data
}

pub(crate) fn apng(width: u32, height: u32) -> Vec<u8> {
    let mut data = png(width, height);
    let mut control = Vec::new();
    png_chunk(&mut control, b"acTL", &[0, 0, 0, 1, 0, 0, 0, 0]);
    data.splice(33..33, control);
    data
}

fn png_chunk(data: &mut Vec<u8>, kind: &[u8; 4], payload: &[u8]) {
    data.extend_from_slice(&u32::try_from(payload.len()).unwrap().to_be_bytes());
    data.extend_from_slice(kind);
    data.extend_from_slice(payload);
    data.extend_from_slice(&[0; 4]);
}

pub(crate) fn jpeg(width: u16, height: u16) -> Vec<u8> {
    let mut data = vec![0xff, 0xd8, 0xff, 0xe0, 0, 4, 0, 0, 0xff, 0xc0, 0, 11, 8];
    data.extend_from_slice(&height.to_be_bytes());
    data.extend_from_slice(&width.to_be_bytes());
    data.extend_from_slice(&[1, 1, 0x11, 0, 0xff, 0xda]);
    data
}

#[test]
fn png_and_jpeg_dimensions_are_detected() {
    assert_eq!(
        detect(&png(256, 1568)).unwrap(),
        (EvaluationImageFormat::Png, (256, 1568))
    );
    assert_eq!(
        detect(&jpeg(1568, 256)).unwrap(),
        (EvaluationImageFormat::Jpeg, (1568, 256))
    );
}

#[test]
fn progressive_jpeg_is_accepted() {
    let mut data = jpeg(512, 768);
    data[9] = 0xc2;
    assert_eq!(
        detect(&data).unwrap(),
        (EvaluationImageFormat::Jpeg, (512, 768))
    );
}

#[test]
fn animated_png_is_rejected() {
    assert!(matches!(
        detect(&apng(512, 512)),
        Err(message)
            if message.contains("animated PNG") && message.contains("acTL")
    ));
}

#[test]
fn png_requires_idat_before_iend() {
    let mut data = png(512, 768);
    data.truncate(33);
    png_chunk(&mut data, b"tEXt", b"image metadata");
    png_chunk(&mut data, b"IEND", &[]);
    assert_eq!(detect(&data), Err("truncated or corrupt PNG chunk header"));
    assert_eq!(
        detect(&png(512, 768)).unwrap(),
        (EvaluationImageFormat::Png, (512, 768))
    );
}

#[test]
fn jpeg_skips_fill_bytes_and_standalone_markers() {
    let mut data = jpeg(512, 768);
    data.splice(8..8, [0xff, 0xff, 0x01, 0xff, 0xd0, 0xff, 0xd7]);
    assert_eq!(
        detect(&data).unwrap(),
        (EvaluationImageFormat::Jpeg, (512, 768))
    );
}

#[test]
fn non_sof_markers_in_the_sof_range_are_skipped() {
    for marker in [0xc4, 0xc8, 0xcc] {
        let mut data = jpeg(512, 768);
        data.splice(8..8, [0xff, marker, 0, 8, 8, 0, 1, 0, 2, 0]);
        assert_eq!(
            detect(&data).unwrap(),
            (EvaluationImageFormat::Jpeg, (512, 768))
        );
    }
}

#[test]
fn every_truncated_png_header_is_rejected() {
    let data = png(512, 512);
    for length in 0..data.len() {
        assert!(detect(&data[..length]).is_err(), "length {length}");
    }
}

#[test]
fn every_truncated_jpeg_header_before_a_complete_sof_is_rejected() {
    let data = jpeg(512, 512);
    for length in 0..data.len() - 2 {
        assert!(detect(&data[..length]).is_err(), "length {length}");
    }
}

#[test]
fn corrupt_png_chunks_are_rejected() {
    let mut wrong_length = png(512, 512);
    wrong_length[11] = 12;
    let mut missing_ihdr = png(512, 512);
    missing_ihdr[12..16].copy_from_slice(b"tEXt");
    let mut oversized_chunk = png(512, 512);
    oversized_chunk[33..37].copy_from_slice(&u32::MAX.to_be_bytes());
    let mut repeated_ihdr = png(512, 512);
    repeated_ihdr[37..41].copy_from_slice(b"IHDR");
    for data in [wrong_length, missing_ihdr, oversized_chunk, repeated_ihdr] {
        assert!(detect(&data).is_err());
    }
}

#[test]
fn corrupt_jpeg_segments_or_missing_sof_are_rejected() {
    let mut short_segment = jpeg(512, 512);
    short_segment[5] = 1;
    let mut long_segment = jpeg(512, 512);
    long_segment[4..6].copy_from_slice(&u16::MAX.to_be_bytes());
    let mut short_sof = jpeg(512, 512);
    short_sof[11] = 7;
    let missing_component_list = b"\xff\xd8\xff\xc0\x00\x08\x08\x02\x00\x02\x00\x01";
    for data in [
        short_segment,
        long_segment,
        short_sof,
        missing_component_list.to_vec(),
        vec![0xff, 0xd8, 0xff, 0xda],
        vec![0xff, 0xd8, 0xff, 0xd9],
        vec![0xff, 0xd8, 0xff, 0xff],
        vec![0xff, 0xd8, 0xff, 0x00],
        vec![0xff, 0xd8, 0, 0xc0],
    ] {
        assert!(detect(&data).is_err());
    }
}

#[test]
fn unknown_bytes_are_rejected() {
    for data in [b"GIF89a".as_slice(), b"RIFF....WEBP", b"not an image"] {
        assert!(matches!(
            detect(data),
            Err(message)
                if message.contains("only PNG or JPEG still images")
        ));
    }
}
