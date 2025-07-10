import torch

class LayerActivations:
    def __init__(self, output_dir: str):
        self.hooks = []
        self._active = False
        self._current_timestep = None
        self.activations = {}  # {timestep: {layer_name: [activations]}}
        self.output_dir = output_dir

    def set_timestep(self, timestep):
        """Set the current timestep for logging."""
        self._current_timestep = timestep
        if self._current_timestep not in self.activations:
            self.activations[self._current_timestep] = {}

    def _get_hook(self, layer_name, timesteps, debug=True):
        print("[DEBUG] Creating hook for layer:", layer_name, "at timesteps:", timesteps) if debug else None
        def hook_fn(module, input, output):
            print("[DEBUG] Hook called for layer:", layer_name, "at timestep:", self._current_timestep) if debug else None
            if self._active and self._current_timestep in timesteps:
                if self._current_timestep in self.activations:
                    if layer_name in self.activations[self._current_timestep]:
                        self.activations[self._current_timestep][layer_name].append(output.detach().cpu())
                    else:
                        self.activations[self._current_timestep][layer_name] = [output.detach().cpu()]
                    if debug:
                        print(f"[DEBUG] Layer: {layer_name}, Timestep: {self._current_timestep}, Output shape: {output.shape}")
        if debug:
            print(f"[DEBUG] Registering hook for layer: {layer_name} at timesteps: {timesteps}")
        return hook_fn

    def _resolve_layer(self, model, layer_path: str):
        """Resolve a dotted layer path like 'transformer_blocks.1.attn.q_proj'."""
        parts = layer_path.split('.')
        layer = model
        for part in parts:
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)
        return layer

    def register(self, model: torch.nn.Module, layer_names: list[str], timesteps: list[int]):
        """
        layer_names: list of strings like 'transformer_blocks.1.attn'
        """
        self.clear()
        print("[DEBUG] Registering hooks for layers:", layer_names, "at timesteps:", timesteps)
        for name in layer_names:
            layer = self._resolve_layer(model, name)
            hook = layer.register_forward_hook(self._get_hook(name, timesteps))
            self.hooks.append(hook)
        self._active = True

    def save(self):
        """
        Save the activations to a file.
        """
        print("[DEBUG] Saving activations to files in directory:", self.output_dir)
        for timestep in self.activations.keys():
            for layer_name, activations in self.activations[timestep].items():
                # Convert list of tensors to a single tensor
                if activations:
                    tensor = torch.stack(activations)
                    file_path = f"{self.output_dir}/activations_{timestep}_{layer_name}.pt"
                    torch.save(tensor, file_path)

    def clear(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
        self._current_timestep = None
        self._active = False

    def get(self):
        return self.activations
    
    def flush(self):
        """
        Save and clear the activations.
        """
        self.save()
        self.clear()
        self._current_timestep = None
