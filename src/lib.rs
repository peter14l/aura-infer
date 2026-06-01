pub mod tensor;
pub mod model;
pub mod loader;

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;
    use crate::model::RMSNorm;

    #[test]
    fn test_matmul() {
        // Input vector: shape [3]
        let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        
        // Weight matrix: shape [2, 3] (2 output neurons, 3 input features)
        let weights = Tensor::new(vec![
            0.2, 0.4, 0.6,  // Weights for output neuron 1
            0.1, 0.3, 0.5   // Weights for output neuron 2
        ], vec![2, 3]);

        let output = input.matmul(&weights);
        
        assert_eq!(output.shape, vec![2]);
        // Calculation for output[0]: (1.0*0.2) + (2.0*0.4) + (3.0*0.6) = 0.2 + 0.8 + 1.8 = 2.8
        // Calculation for output[1]: (1.0*0.1) + (2.0*0.3) + (3.0*0.5) = 0.1 + 0.6 + 1.5 = 2.2
        assert!((output.data[0] - 2.8).abs() < 1e-5);
        assert!((output.data[1] - 2.2).abs() < 1e-5);
    }

    #[test]
    fn test_rmsnorm() {
        let norm = RMSNorm::new(4, 1e-5);
        let input = Tensor::new(vec![1.0, -1.0, 2.0, -2.0], vec![4]);
        let output = norm.forward(&input);
        
        // RMS of [1, -1, 2, -2] -> squares: [1, 1, 4, 4] -> sum: 10 -> mean: 2.5 -> sqrt(2.5) ~= 1.5811
        // Expected output ~ [0.6324, -0.6324, 1.2649, -1.2649]
        assert_eq!(output.shape, vec![4]);
        assert!(output.data[0] > 0.0);
        assert!(output.data[1] < 0.0);
    }
}
