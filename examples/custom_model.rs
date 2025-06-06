use relbench::{
    get_dataset, get_task,
    model::{Model, ModelConfig},
    Result,
};
use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

/// Custom Graph Neural Network model
struct CustomGNN {
    embedding: Linear,
    conv1: Linear,
    conv2: Linear,
    output: Linear,
}

impl CustomGNN {
    fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let input_size = config.input_size;
        let output_size = config.output_size;
        
        Ok(Self {
            embedding: Linear::new(
                vb.pp("embedding"),
                input_size,
                hidden_size,
                true,
            )?,
            conv1: Linear::new(
                vb.pp("conv1"),
                hidden_size,
                hidden_size,
                true,
            )?,
            conv2: Linear::new(
                vb.pp("conv2"),
                hidden_size,
                hidden_size,
                true,
            )?,
            output: Linear::new(
                vb.pp("output"),
                hidden_size,
                output_size,
                true,
            )?,
        })
    }
}

impl Model for CustomGNN {
    fn name(&self) -> &str {
        "custom_gnn"
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Node feature embedding
        let h = self.embedding.forward(x)?;
        
        // Graph convolutions
        let h = self.conv1.forward(&h)?.relu()?;
        let h = self.conv2.forward(&h)?.relu()?;
        
        // Output layer
        let out = self.output.forward(&h)?;
        
        Ok(out)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load dataset and task
    let dataset = get_dataset("rel-amazon", true)?;
    let task = get_task("rel-amazon", "user-churn", true)?;
    
    // Get data
    let train = task.get_train_table()?;
    let val = task.get_val_table()?;
    let test = task.get_test_table(true)?;
    
    // Model configuration
    let config = ModelConfig {
        input_size: 64,
        hidden_size: 128,
        output_size: 1,
        num_layers: 2,
        dropout: 0.1,
        learning_rate: 0.001,
    };
    
    // Initialize model
    let vb = VarBuilder::zeros(Device::Cpu);
    let model = CustomGNN::new(&config, vb)?;
    
    // Training loop (simplified)
    for epoch in 0..10 {
        // Forward pass
        let x = Tensor::zeros((train.len(), config.input_size), Device::Cpu)?;
        let predictions = model.forward(&x)?;
        
        println!("Epoch {}: predictions shape {:?}", epoch, predictions.shape());
    }
    
    // Evaluation
    let x_test = Tensor::zeros((test.len(), config.input_size), Device::Cpu)?;
    let predictions = model.forward(&x_test)?;
    let predictions = predictions.squeeze(1)?.to_vec1::<f32>()?;
    
    // Compute metrics
    let metrics = task.evaluate(&predictions, None)?;
    println!("\nTest metrics:");
    for (name, value) in metrics.iter() {
        println!("- {}: {:.4}", name, value);
    }
    
    Ok(())
} 