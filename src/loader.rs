use std::fs::File;
use std::io::{Read, Result, Error, ErrorKind};
use crate::tensor::Tensor;

/// A parser for .safetensors or .bin model files.
/// In a production engine, this would memory-map (mmap) the file instead of reading it into RAM.
pub struct ModelLoader {
    filepath: String,
}

impl ModelLoader {
    pub fn new(filepath: &str) -> Self {
        ModelLoader {
            filepath: filepath.to_string(),
        }
    }

    /// Simulates reading a specific tensor (e.g., "layers.0.attention.wq.weight") from disk.
    /// Converts raw Little-Endian bytes directly into f32 floating point numbers.
    pub fn load_tensor(&self, name: &str, expected_shape: Vec<usize>) -> Result<Tensor> {
        // In a real .safetensors implementation, we would first parse the JSON header 
        // to find the exact byte offsets for `name`. For this PoC, we will simulate reading
        // raw bytes if the file exists, or return a dummy tensor if it doesn't.
        
        let mut file = match File::open(&self.filepath) {
            Ok(f) => f,
            Err(_) => {
                // For demonstration, if no model file exists, return a dummy initialized tensor
                // so the engine can still run mathematically.
                return Ok(Tensor::new(vec![0.01; expected_shape.iter().product()], expected_shape));
            }
        };

        let num_elements: usize = expected_shape.iter().product();
        let num_bytes = num_elements * 4; // 4 bytes per f32
        
        let mut buffer = vec![0u8; num_bytes];
        file.read_exact(&mut buffer)?;

        // Convert raw bytes to f32
        let mut data = Vec::with_capacity(num_elements);
        for i in 0..num_elements {
            let start = i * 4;
            let bytes = [buffer[start], buffer[start+1], buffer[start+2], buffer[start+3]];
            data.push(f32::from_le_bytes(bytes));
        }

        Ok(Tensor::new(data, expected_shape))
    }
}
