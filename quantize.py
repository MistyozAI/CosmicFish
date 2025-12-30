import torch
import os
from model import CosmicFish, CosmicConfig
from torch.serialization import add_safe_globals

# Add safe globals for loading
add_safe_globals([CosmicConfig])

def convert_to_fp16(input_path, output_path):
    print("Loading original model...")
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
    
    # Get config
    if 'cosmicconf' in checkpoint:
        config = checkpoint['cosmicconf']
    elif 'config' in checkpoint:
        config = checkpoint['config'] 
    else:
        print("No config found!")
        return
    
    # Create model
    model = CosmicFish(config)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    
    print(f"Original model: {model.get_num_params() / 1e6:.2f}M parameters")
    
    # Convert to half precision
    model = model.half()
    
    # Save
    new_checkpoint = {
        'model_state_dict': model.state_dict(),
        'cosmicconf': config,
        'precision': 'float16'
    }
    
    torch.save(new_checkpoint, output_path)
    
    # Size comparison
    original_size = os.path.getsize(input_path) / (1024*1024)
    new_size = os.path.getsize(output_path) / (1024*1024)
    
    print(f"✅ Original: {original_size:.1f}MB")
    print(f"✅ Float16: {new_size:.1f}MB ({100-new_size/original_size*100:.1f}% reduction)")
    print(f"✅ Saved to: {output_path}")

if __name__ == "__main__":
    convert_to_fp16(
        "/home/akhil/Downloads/best_calibrated (1).pt",
        "CF300M_new.pt"
    )
