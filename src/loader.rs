use std::fs::File;
use std::io::Result;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use memmap2::MmapOptions;

/// A parser for .safetensors model files.
/// Uses memory mapping (mmap) to map the file directly into RAM for zero-copy loading.
pub struct ModelLoader<'a> {
    mmap: memmap2::Mmap,
    _marker: std::marker::PhantomData<&'a ()>, // Ties the lifetime of the safetensors to the mmap
}

impl<'a> ModelLoader<'a> {
    pub fn new(filepath: &str) -> Result<Self> {
        let file = File::open(filepath)?;
        // Memory map the file (Zero-copy loading of multi-gigabyte models)
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        
        Ok(ModelLoader {
            mmap,
            _marker: std::marker::PhantomData,
        })
    }

    /// Loads a specific tensor from the safetensors file.
    pub fn load_tensor(&self, name: &str) -> std::result::Result<Tensor, String> {
        // Parse the header
        let st = SafeTensors::deserialize(&self.mmap).map_err(|e| format!("Safetensors error: {:?}", e))?;
        
        let tensor_view = st.tensor(name).map_err(|e| format!("Tensor not found: {:?}", e))?;
        
        // Convert shape from usize array to Vec
        let shape: Vec<usize> = tensor_view.shape().iter().copied().collect();
        let num_elements = shape.iter().product();
        let data_bytes = tensor_view.data();
        
        // Convert raw bytes to f32 (Assuming Little Endian Float32)
        let mut data = Vec::with_capacity(num_elements);
        for i in 0..num_elements {
            let start = i * 4;
            let bytes = [
                data_bytes[start], 
                data_bytes[start+1], 
                data_bytes[start+2], 
                data_bytes[start+3]
            ];
            data.push(f32::from_le_bytes(bytes));
        }

        Ok(Tensor::new(data, shape))
    }
}
