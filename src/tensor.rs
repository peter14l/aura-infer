/// A basic 1D or 2D Tensor for Neural Network calculations.
/// In a production environment, this would wrap raw memory pointers or map to GPU memory.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Creates a new Tensor filled with zeros.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Tensor {
            data: vec![0.0; size],
            shape,
        }
    }

    /// Creates a new Tensor from existing data.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(data.len(), shape.iter().product(), "Data length does not match shape");
        Tensor { data, shape }
    }

    /// The absolute core of AI: Matrix Multiplication (Linear Layer).
    /// Computes Output = Input x Weights.
    /// Assuming `self` is a 1D vector (shape [cols]) and `weights` is a 2D matrix (shape [rows, cols]).
    pub fn matmul(&self, weights: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 1, "Input must be 1D for this naive matmul");
        assert_eq!(weights.shape.len(), 2, "Weights must be 2D");
        
        let in_features = weights.shape[1];
        let out_features = weights.shape[0];
        
        assert_eq!(self.shape[0], in_features, "Dimension mismatch for matmul");

        let mut output = Tensor::zeros(vec![out_features]);

        // This is the O(N^2) operation that GPUs accelerate. 
        // We write the bare-metal CPU implementation here.
        for i in 0..out_features {
            let mut sum = 0.0;
            for j in 0..in_features {
                // Accessing weights[i][j] in a flat 1D vector
                sum += self.data[j] * weights.data[i * in_features + j];
            }
            output.data[i] = sum;
        }

        output
    }

    /// Calculates the Softmax of the tensor (in-place or returning new). 
    /// Used in Self-Attention to turn raw scores into probabilities that sum to 1.0.
    pub fn softmax(&self) -> Tensor {
        assert_eq!(self.shape.len(), 1, "Softmax only implemented for 1D tensors in this PoC");
        let mut max_val = f32::NEG_INFINITY;
        
        // Find max for numerical stability (prevents NaN when doing exp)
        for &val in &self.data {
            if val > max_val { max_val = val; }
        }

        let mut out = Tensor::zeros(self.shape.clone());
        let mut sum = 0.0;

        for i in 0..self.data.len() {
            out.data[i] = (self.data[i] - max_val).exp();
            sum += out.data[i];
        }

        for i in 0..self.data.len() {
            out.data[i] /= sum;
        }

        out
    }

    /// Scales the tensor in-place by a constant factor.
    pub fn scale_(&mut self, factor: f32) {
        for i in 0..self.data.len() {
            self.data[i] *= factor;
        }
    }

    /// Adds another tensor into this one (element-wise).
    pub fn add_(&mut self, other: &Tensor) {
        assert_eq!(self.shape, other.shape, "Shapes must match for addition");
        for i in 0..self.data.len() {
            self.data[i] += other.data[i];
        }
    }
}
