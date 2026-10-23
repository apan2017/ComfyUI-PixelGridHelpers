import torch
import re

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


OPAQUE_IS_1 = "opaque_is_1"      # 1 = drawn/opaque  (this pack's own convention)
OPAQUE_IS_0 = "opaque_is_0"      # 1 = clear/transparent (ComfyUI LoadImage's MASK!)
MASK_MODES = ["auto", OPAQUE_IS_1, OPAQUE_IS_0]


def _resolve_alpha(image, mask=None, chroma_key="", chroma_tolerance=0.10,
                   mask_mode="auto", info=None):
    """Build a [B,H,W] float alpha (1=opaque, 0=transparent) from any source.

    Priority: MASK input > chroma_key hex > everything opaque.

    `mask_mode` matters because ComfyUI's own mask convention is INVERTED
    relative to naive intuition: `LoadImage` emits `mask = 1 - alpha`, i.e.
    **1 means transparent** (it is an inpainting mask). Masks produced by the
    nodes in this pack use the opposite, natural convention (1 = opaque).

      auto       -> decide from the image border (a sprite/tile is transparent
                    at the frame edge), reported back through `info`
      opaque_is_1 / opaque_is_0 -> force one interpretation

    The mask is resampled to the image size when they disagree (happens when the
    IMAGE was scaled but the MASK was not) and broadcast across the batch.
    """
    B, H, W, _ = image.shape
    if mask is not None:
        m = mask.to(device=image.device, dtype=torch.float32)
        if m.dim() == 4:            # [B,H,W,1] from some packs
            m = m[..., 0]
        elif m.dim() == 2:          # bare [H,W]
            m = m.unsqueeze(0)
        if tuple(m.shape[-2:]) != (H, W):
            m = torch.nn.functional.interpolate(
                m.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
            ).squeeze(1)
        if m.shape[0] < B:
            m = torch.cat([m, m[-1:].expand(B - m.shape[0], H, W)], dim=0)
        elif m.shape[0] > B:
            m = m[:B]
        m = m.clamp(0.0, 1.0)

        mode = mask_mode
        if mode == "auto":
            # Border heuristic: for a sprite/tile the frame edge is background.
            # Pick whichever interpretation makes that edge transparent.
            border = torch.cat([m[0, 0, :], m[0, -1, :], m[0, :, 0], m[0, :, -1]])
            mode = OPAQUE_IS_0 if float(border.mean()) > 0.5 else OPAQUE_IS_1
        if mode == OPAQUE_IS_0:
            m = 1.0 - m

        if info is not None:
            info["mask_source"] = "mask input"
            info["mask_mode"] = mode
            if mode != mask_mode:
                info["mask_mode"] += " (auto)"
        return m.clamp(0.0, 1.0)

    if chroma_key and chroma_key.strip():
        key = torch.tensor(
            list(hex_to_rgb(chroma_key.strip())),
            dtype=torch.float32, device=image.device,
        ) / 255.0
        px = image[..., :3].reshape(-1, 3)
        dist = torch.sqrt(((px - key) ** 2).sum(dim=1))
        if info is not None:
            info["mask_source"] = f"chroma_key {chroma_key.strip()} tol {chroma_tolerance}"
            info["mask_mode"] = "-"
        return (dist > chroma_tolerance).to(torch.float32).reshape(B, H, W)

    if info is not None:
        info["mask_source"] = "none -> assumed all opaque"
        info["mask_mode"] = "-"
    return torch.ones((B, H, W), dtype=torch.float32, device=image.device)


def _edge_extend_rgb(rgb, opaque):
    """Repaint every NON-opaque pixel with its nearest opaque neighbour's color.

    `rgb` is a float32 numpy [H,W,3]; `opaque` is a bool [H,W]. The nearest
    match is exact (Euclidean distance transform), so the transparent region
    ends up holding real artwork colors instead of a flat fill. That is what
    makes the image safe to resample: there is no foreign fill color left for a
    smoothing filter to smear into a halo.
    """
    import scipy.ndimage as ndi

    if opaque.all() or not opaque.any():
        return rgb

    # scipy treats NONZERO input as the feature and returns the nearest ZERO,
    # so we feed it the TRANSPARENT mask: its zeros are the opaque pixels and
    # the returned indices point at the nearest one.
    _, inds = ndi.distance_transform_edt(~opaque, return_indices=True)
    nearest = rgb[inds[0], inds[1]]
    out = rgb.copy()
    out[~opaque] = nearest[~opaque]
    return out

class PixelGrid_CreateSolidColorImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "hex_code": ("STRING", {"default": "#FF0000"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "create_image"
    CATEGORY = "Pixel Grid Helpers"

    def create_image(self, width, height, hex_code):
        match = re.fullmatch(r"#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})", hex_code.strip())
        if match is None:
            raise ValueError(
                f"Invalid HEX color {hex_code!r}; use 3 or 6 hexadecimal digits."
            )

        digits = match.group(1)
        if len(digits) == 3:
            digits = "".join(digit * 2 for digit in digits)
        rgb = [int(digits[index:index + 2], 16) for index in (0, 2, 4)]
        color = torch.tensor(rgb, dtype=torch.float32).div_(255).view(1, 1, 1, 3)
        image = color.expand(1, height, width, 3).clone()
        return (image,)

class PixelGrid_KMeans:
    """Node 1: Enforce Max N Colors (K-Means Quantization)"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_colors": ("INT", {"default": 16, "min": 1, "max": 256}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("quantized_image", "hex_palette")
    FUNCTION = "quantize"
    CATEGORY = "Pixel Grid Helpers"

    def quantize(self, image, max_colors):
        # image: [B, H, W, 3]
        # Flatten: [N, 3]
        pixels = image.reshape(-1, 3)
        
        # Simple K-Means implementation in Torch
        # Initialize centroids randomly from existing pixels
        num_pixels = pixels.shape[0]
        indices = torch.randperm(num_pixels)[:max_colors]
        centroids = pixels[indices]

        for _ in range(10): # 10 iterations is usually enough for pixel art
            # Calculate distances [Pixels, Centroids]
            dists = torch.cdist(pixels, centroids)
            # Assign to nearest
            labels = torch.argmin(dists, dim=1)
            # Update centroids
            new_centroids = []
            for i in range(max_colors):
                mask = (labels == i)
                if mask.any():
                    new_centroids.append(pixels[mask].mean(dim=0))
                else:
                    # If a centroid has no pixels, re-init it random
                    random_idx = torch.randint(0, num_pixels, (1,))
                    new_centroids.append(pixels[random_idx].squeeze())
            centroids = torch.stack(new_centroids)

        # Final assignment
        dists = torch.cdist(pixels, centroids)
        labels = torch.argmin(dists, dim=1)
        
        # Reconstruct
        quantized = centroids[labels].reshape(image.shape)
        
        # Generate Palette String
        # Convert 0-1 float rgb to 0-255 int
        palette_rgb = (centroids * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        hex_list = [rgb_to_hex(c) for c in palette_rgb]
        hex_string = ", ".join(hex_list)

        return (quantized, hex_string)

class PixelGrid_ApplyPalette:
    """Node 2: Enforce Provided Color Palette"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "hex_palette": ("STRING", {"multiline": True, "default": "#000000, #FFFFFF"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("quantized_image", "hex_palette_out")
    FUNCTION = "apply_palette"
    CATEGORY = "Pixel Grid Helpers"

    def apply_palette(self, image, hex_palette):
        # Parse hex string
        hex_codes = [x.strip() for x in hex_palette.split(',') if x.strip()]
        if not hex_codes:
            return (image, hex_palette)

        # Create Palette Tensor [K, 3]
        rgb_list = [hex_to_rgb(h) for h in hex_codes]
        palette = torch.tensor(rgb_list, dtype=torch.float32, device=image.device) / 255.0

        # Flatten image
        pixels = image.reshape(-1, 3)
        
        # Find nearest color
        dists = torch.cdist(pixels, palette) # [N_pixels, N_palette]
        labels = torch.argmin(dists, dim=1)
        
        # Map
        new_pixels = palette[labels]
        result = new_pixels.reshape(image.shape)

        return (result, hex_palette)

class PixelGrid_MergeSimilar:
    """Node 3: Merge Similar Colors by Threshold"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("merged_image", "hex_palette")
    FUNCTION = "merge_colors"
    CATEGORY = "Pixel Grid Helpers"

    def merge_colors(self, image, threshold):
        # 1. Get Unique Colors
        pixels = image.reshape(-1, 3)
        unique_colors, inverse_indices = torch.unique(pixels, dim=0, return_inverse=True)
        
        # 2. Greedy Merging
        # We will iterate and merge colors that are close
        final_palette = [] # List of tensors
        
        # We use a mask to track processed colors
        processed = torch.zeros(unique_colors.shape[0], dtype=torch.bool, device=image.device)
        
        for i in range(unique_colors.shape[0]):
            if processed[i]:
                continue
            
            current_color = unique_colors[i]
            
            # Calculate distance to all unprocessed colors
            dists = torch.norm(unique_colors - current_color, dim=1)
            
            # Find close colors (including itself)
            mask = (dists <= threshold) & (~processed)
            
            # Average these colors
            group = unique_colors[mask]
            merged_color = group.mean(dim=0)
            
            final_palette.append(merged_color)
            
            # Mark as processed
            processed = processed | mask
            
            # Update the unique_colors array for the remapping step
            # (We overwrite the original unique colors with the new merged one so mapping works)
            unique_colors[mask] = merged_color

        # 3. Reconstruct
        # inverse_indices points to the original unique_colors table, which we just updated in place
        new_pixels = unique_colors[inverse_indices]
        result = new_pixels.reshape(image.shape)
        
        # 4. Generate Palette String
        palette_tensor = torch.stack(final_palette)
        palette_rgb = (palette_tensor * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        hex_list = [rgb_to_hex(c) for c in palette_rgb]
        
        return (result, ", ".join(hex_list))

class PixelGrid_Analyze:
    """Node 4: Analyze Color Palette"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { "image": ("IMAGE",), }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("hex_palette",)
    FUNCTION = "analyze"
    CATEGORY = "Pixel Grid Helpers"

    def analyze(self, image):
        pixels = image.reshape(-1, 3)
        unique_colors = torch.unique(pixels, dim=0)
        
        palette_rgb = (unique_colors * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        hex_list = [rgb_to_hex(c) for c in palette_rgb]
        return (", ".join(hex_list),)

class PixelGrid_PaletteToImage:
    """Node 5: Palette to Image Swatches"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { "hex_palette": ("STRING", {"multiline": True, "default": "#FF0000, #00FF00"}), }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("swatch_image",)
    FUNCTION = "generate"
    CATEGORY = "Pixel Grid Helpers"

    def generate(self, hex_palette):
        hex_codes = [x.strip() for x in hex_palette.split(',') if x.strip()]
        if not hex_codes:
             # Return black 24x24 if empty
             return (torch.zeros((1, 24, 24, 3)),)

        swatches = []
        for h in hex_codes:
            rgb = hex_to_rgb(h)
            # Create [24, 24, 3] tensor
            # Normalize to 0-1
            tensor_color = torch.tensor(rgb, dtype=torch.float32) / 255.0
            patch = tensor_color.view(1, 1, 3).repeat(24, 24, 1)
            swatches.append(patch)
        
        # Stack vertically [H*Count, W, 3]
        full_image = torch.cat(swatches, dim=0)
        # Add batch dim [1, H, W, 3]
        full_image = full_image.unsqueeze(0)
        
        return (full_image,)

# Re-including the previous node requested
class GridMedianFixer:
    """Median-reduce an image using the most coherent pixel-grid phase."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "grid_size": ("INT", {"default": 6, "min": 1, "max": 128}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("downscaled_image", "original_size_image")
    FUNCTION = "process_grid"
    CATEGORY = "Pixel Grid Helpers"

    @staticmethod
    def _phase_downscaled(image, grid_size, y_offset, x_offset):
        batch, height, width, channels = image.shape
        rows = max((height - y_offset) // grid_size, 0)
        columns = max((width - x_offset) // grid_size, 0)
        end_y = y_offset + rows * grid_size
        end_x = x_offset + columns * grid_size
        cropped = image[:, y_offset:end_y, x_offset:end_x, :]

        blocks = cropped.reshape(
            batch, rows, grid_size, columns, grid_size, channels
        ).permute(0, 1, 3, 2, 4, 5)
        blocks = blocks.reshape(
            batch, rows, columns, grid_size * grid_size, channels
        )
        return torch.median(blocks, dim=3).values

    @classmethod
    def _phase_outputs(cls, image, grid_size, y_offset, x_offset):
        downscaled = cls._phase_downscaled(
            image, grid_size, y_offset, x_offset
        )
        upscaled = downscaled.repeat_interleave(grid_size, dim=1).repeat_interleave(
            grid_size, dim=2
        )
        return downscaled, upscaled

    @staticmethod
    def _common_phase_bounds(length, grid_size):
        valid_offsets = [
            offset
            for offset in range(grid_size)
            if length - offset >= grid_size
        ]
        if not valid_offsets:
            return 0, 0

        start = valid_offsets[-1]
        end = min(
            offset + ((length - offset) // grid_size) * grid_size
            for offset in valid_offsets
        )
        return start, end

    @staticmethod
    def _phase_error(
        image,
        grid_size,
        y_offset,
        x_offset,
        score_top,
        score_bottom,
        score_left,
        score_right,
    ):
        batch, _, _, channels = image.shape
        first_row = (score_top - y_offset) // grid_size
        last_row = (score_bottom - 1 - y_offset) // grid_size + 1
        first_column = (score_left - x_offset) // grid_size
        last_column = (score_right - 1 - x_offset) // grid_size + 1

        block_top = y_offset + first_row * grid_size
        block_bottom = y_offset + last_row * grid_size
        block_left = x_offset + first_column * grid_size
        block_right = x_offset + last_column * grid_size
        rows = last_row - first_row
        columns = last_column - first_column
        score_pixels = image[:, block_top:block_bottom, block_left:block_right, :]
        blocks = score_pixels.reshape(
            batch, rows, grid_size, columns, grid_size, channels
        ).permute(0, 1, 3, 2, 4, 5)
        blocks = blocks.reshape(
            batch, rows, columns, grid_size * grid_size, channels
        )
        score_medians = torch.median(blocks, dim=3).values
        aligned_region = score_medians.repeat_interleave(
            grid_size, dim=1
        ).repeat_interleave(grid_size, dim=2)

        local_top = score_top - block_top
        local_left = score_left - block_left
        region_height = score_bottom - score_top
        region_width = score_right - score_left
        aligned_region = aligned_region[
            :, local_top:local_top + region_height,
            local_left:local_left + region_width, :
        ]
        source_region = image[:, score_top:score_bottom, score_left:score_right, :]
        error = source_region.to(torch.float32) - aligned_region.to(torch.float32)
        return error.abs().mean()

    def process_grid(self, image, grid_size):
        if grid_size < 1:
            raise ValueError("grid_size must be at least 1")

        _, height, width, _ = image.shape
        with torch.no_grad():
            if height < grid_size or width < grid_size:
                return self._phase_outputs(image, grid_size, 0, 0)

            score_top, score_bottom = self._common_phase_bounds(height, grid_size)
            score_left, score_right = self._common_phase_bounds(width, grid_size)
            phases = []
            scores = []

            y_offsets = range(min(grid_size, height - grid_size + 1))
            x_offsets = range(min(grid_size, width - grid_size + 1))
            for y_offset in y_offsets:
                for x_offset in x_offsets:
                    score = self._phase_error(
                        image,
                        grid_size,
                        y_offset,
                        x_offset,
                        score_top,
                        score_bottom,
                        score_left,
                        score_right,
                    )
                    phases.append((y_offset, x_offset))
                    scores.append(score)

            best_index = torch.argmin(torch.stack(scores)).item()
            best_y, best_x = phases[best_index]
            return self._phase_outputs(image, grid_size, best_y, best_x)


class PixelGrid_KMeansSkipTransparent:
    """Node: K-Means Quantize that SKIPS transparent pixels.

    The original PixelGrid_KMeans operates on RGB only (ComfyUI IMAGE has no
    alpha), so the colored fringe left behind by background removal (e.g.
    magenta 255,0,255) gets quantized too and bleeds into the palette as
    purple/magenta "noise". This node excludes transparent pixels from the
    clustering, so the background can never pollute the palette.

    Transparent pixels still need SOME value in an RGB tensor. `fill="edge"`
    (default) edge-pads them with the nearest opaque palette color, so nothing
    foreign is left behind -- that is what keeps a later smooth resize from
    smearing a black/white halo, and it is why the preview no longer shows a
    black background. The flat options (dominant/black/magenta) are still there
    if you explicitly want a keyable solid color.

    Outputs:
      - quantized_image (RGB): quantized artwork, transparent area edge-padded
        (or flat-filled when fill != "edge"); an RGB tensor cannot hold alpha,
        the real alpha is in `mask`.
      - hex_palette (STRING): opaque colors only, de-duplicated.
      - mask (MASK): the alpha (1=opaque, 0=transparent) for re-compositing.

    `mask_mode` covers the mask-convention trap: ComfyUI's own `LoadImage` MASK
    is `1 - alpha` (1 = transparent), while this pack emits 1 = opaque. "auto"
    decides from the image border, which is transparent for a sprite/tile.

    ComfyUI's IMAGE type is RGB-native (a 4-channel IMAGE is not standard and is
    rejected/noised downstream), so this node does NOT try to emit RGBA. The
    alpha rides along in `mask`; feed `quantized_image` + `mask` into
    `Pixel Grid: Restore Transparency` to key any flat fill back out.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_colors": ("INT", {"default": 32, "min": 1, "max": 256}),
            },
            "optional": {
                "mask": ("MASK",),
                "mask_mode": (MASK_MODES, {"default": "auto"}),
                "chroma_key": ("STRING", {"multiline": False, "default": ""}),
                "chroma_tolerance": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fill": (["edge", "dominant", "black", "magenta"], {"default": "edge"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("quantized_image", "hex_palette", "mask")
    FUNCTION = "quantize"
    CATEGORY = "Pixel Grid Helpers"

    @staticmethod
    def _to_opaque_mask(image, mask, chroma_key, chroma_tolerance,
                        mask_mode="auto", info=None):
        alpha = _resolve_alpha(image, mask, chroma_key, chroma_tolerance,
                               mask_mode, info)
        return alpha.reshape(-1) > 0.5

    def quantize(self, image, max_colors, mask=None, mask_mode="auto",
                 chroma_key="", chroma_tolerance=0.10, fill="edge"):
        B, H, W, _ = image.shape
        pixels = image.reshape(-1, 3)            # [N, 3]
        info = {}
        opaque = self._to_opaque_mask(image, mask, chroma_key, chroma_tolerance,
                                      mask_mode, info)
        opaque_px = pixels[opaque]

        print(f"[Pixel Grid] Quantize Skip Transparent: {W}x{H} | "
              f"transparency from {info.get('mask_source')} | "
              f"mask_mode {info.get('mask_mode')} | "
              f"opaque {100.0 * float(opaque.sum()) / max(1, opaque.numel()):.1f}%")

        # Fallback: nothing opaque -> just pass through
        if opaque_px.shape[0] == 0:
            out_mask = opaque.reshape(B, H, W).to(torch.float32)
            return (image, "", out_mask)

        # K-Means (10 iters) on OPAQUE pixels only
        n = opaque_px.shape[0]
        idx = torch.randperm(n)[:max_colors]
        centroids = opaque_px[idx].clone()
        for _ in range(10):
            d = torch.cdist(opaque_px, centroids)
            labels = torch.argmin(d, dim=1)
            new_c = []
            for i in range(max_colors):
                mm = (labels == i)
                if mm.any():
                    new_c.append(opaque_px[mm].mean(dim=0))
                else:
                    ridx = torch.randint(0, n, (1,))
                    new_c.append(opaque_px[ridx].squeeze())
            centroids = torch.stack(new_c)

        d = torch.cdist(opaque_px, centroids)
        labels = torch.argmin(d, dim=1)

        # Reconstruct output RGB
        out = pixels.clone()
        out[opaque] = centroids[labels]

        if fill == "edge":
            # Edge-pad the transparent area with the nearest OPAQUE (already
            # quantized) color instead of a flat fill. Nothing foreign is left
            # in the RGB tensor, so a later bilinear/lanczos resize cannot smear
            # a black/white halo into the silhouette, and no "black background"
            # shows up in preview either.
            out3 = out.view(B, H, W, 3)
            om = opaque.reshape(B, H, W)
            for b in range(B):
                fixed = _edge_extend_rgb(
                    out3[b].cpu().numpy(), om[b].cpu().numpy())
                out3[b] = torch.from_numpy(fixed).to(device=out.device, dtype=out.dtype)
        else:
            if fill == "black":
                fcol = torch.tensor([0.0, 0.0, 0.0], device=image.device)
            elif fill == "magenta":
                fcol = torch.tensor([1.0, 0.0, 1.0], device=image.device)
            else:  # dominant -> most populated centroid (reuses an existing slot)
                counts = torch.bincount(labels, minlength=max_colors)
                fcol = centroids[counts.argmax()].clone()
            out[~opaque] = fcol

        out_img = out.view(image.shape)

        # Palette = opaque centroids only (no purple noise), de-duplicated
        palette_rgb = (centroids * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        unique_pal = sorted({tuple(int(v) for v in c) for c in palette_rgb})
        hex_string = ", ".join(rgb_to_hex(list(c)) for c in unique_pal)

        out_mask = opaque.reshape(B, H, W).to(torch.float32)
        return (out_img, hex_string, out_mask)


class PixelGrid_Defringe:
    """Node: remove colored/white fringing trapped in transparent edges.

    Background removal usually sets alpha to transparent but leaves the OLD
    background RGB sitting inside those pixels (white, magenta chroma-key
    residue, etc.). When the image is later resized with a smoothing filter,
    that residue smears into a visible halo.

    This node replaces every NON-opaque pixel's RGB with the color of the
    NEAREST opaque pixel (found via an exact Euclidean distance transform), so
    there is no color mismatch left to bleed. The alpha/mask is left untouched,
    so genuine anti-aliasing is preserved and only the fringe *color* is
    corrected. Transparent pixels stay invisible either way.

    Works for:
      - a MASK input (e.g. LoadImage's mask), or
      - a chroma_key hex + tolerance (treat near-key colors as transparent), or
      - nothing supplied -> passes through unchanged.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "mask": ("MASK",),
                "mask_mode": (MASK_MODES, {"default": "auto"}),
                "chroma_key": ("STRING", {"multiline": False, "default": ""}),
                "chroma_tolerance": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "alpha_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("defringed_image", "mask")
    FUNCTION = "defringe"
    CATEGORY = "Pixel Grid Helpers"

    @staticmethod
    def _alpha(image, mask, chroma_key, chroma_tolerance, mask_mode="auto"):
        return _resolve_alpha(image, mask, chroma_key, chroma_tolerance, mask_mode)

    def defringe(self, image, mask=None, mask_mode="auto", chroma_key="",
                 chroma_tolerance=0.10, alpha_threshold=0.5):
        B, H, W, _ = image.shape
        alpha = _resolve_alpha(image, mask, chroma_key, chroma_tolerance, mask_mode)
        out = image[..., :3].clone()

        for b in range(B):
            opaque = alpha[b].cpu().numpy() >= alpha_threshold
            fixed = _edge_extend_rgb(image[b, :, :, :3].cpu().numpy(), opaque)
            out[b] = torch.from_numpy(fixed).to(device=out.device, dtype=out.dtype)

        return (out, alpha)


class PixelGrid_RestoreTransparency:
    """Node: undo the quantizer's fill color and hand back RGB + alpha.

    Why this exists
    ---------------
    A ComfyUI IMAGE is a 3-channel RGB tensor -- it physically cannot carry
    alpha. So `Pixel Grid: Quantize Skip Transparent` has to paint the
    transparent region with SOME solid color (its `fill`: dominant/black/
    magenta). That paint is the "black background" you see, and if the image is
    then resized with a smoothing filter, the paint smears into the silhouette
    and becomes a black/colored halo around the sprite.

    This is the mid-pipeline counterpart: it drops the paint back out and
    returns a normal IMAGE + MASK pair, so you can keep scaling, merging and
    quantizing, and only composite the alpha at the very end.

    The RGB it emits is *edge-padded*: every pixel outside the mask is repainted
    with the color of its nearest opaque neighbour. Nothing of the fill color
    survives, so resampling with bilinear / bicubic / lanczos cannot drag a
    foreign color into the edge. The MASK output is the alpha, unchanged.

    Outputs:
      image (IMAGE): fill-free RGB, safe to resize.
      mask  (MASK) : alpha, same values as the input mask (1=opaque, 0=clear).
      report (STRING): what actually happened -- see it in the console too.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "mask": ("MASK",),
                "mask_mode": (MASK_MODES, {"default": "auto"}),
                "chroma_key": ("STRING", {"multiline": False, "default": ""}),
                "chroma_tolerance": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "alpha_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "report")
    FUNCTION = "restore"
    CATEGORY = "Pixel Grid Helpers"

    def restore(self, image, mask=None, mask_mode="auto", chroma_key="",
                chroma_tolerance=0.10, alpha_threshold=0.5):
        import numpy as np

        B, H, W, _ = image.shape
        info = {}
        alpha = _resolve_alpha(image, mask, chroma_key, chroma_tolerance,
                               mask_mode, info)
        out = image[..., :3].clone()

        repainted = 0
        fill_guess = None
        for b in range(B):
            opaque = alpha[b].cpu().numpy() >= alpha_threshold
            if fill_guess is None and not opaque.all() and opaque.any():
                y, x = np.argwhere(~opaque)[0]
                c = image[b, y, x, :3].cpu().numpy() * 255.0
                fill_guess = rgb_to_hex(c)
            repainted += int((~opaque).sum())
            fixed = _edge_extend_rgb(image[b, :, :, :3].cpu().numpy(), opaque)
            out[b] = torch.from_numpy(fixed).to(device=out.device, dtype=out.dtype)

        total = max(1, B * H * W)
        a = alpha.detach().cpu().float()
        report = (
            f"{W}x{H} | transparency from {info.get('mask_source')} "
            f"| mask_mode {info.get('mask_mode')} "
            f"| alpha min {float(a.min()):.2f} max {float(a.max()):.2f} "
            f"mean {float(a.mean()):.2f} "
            f"| opaque {100.0 * (total - repainted) / total:.1f}% "
            f"| repainted {100.0 * repainted / total:.1f}%"
        )
        if fill_guess:
            report += f" | removed fill ~{fill_guess}"

        if mask is None and not (chroma_key and chroma_key.strip()):
            report += " | WARNING: no mask/chroma_key -> nothing to restore"
            print(f"[Pixel Grid] Restore Transparency: {report}")
            print("[Pixel Grid]   -> connect Quantize's `mask` output (1=opaque) "
                  "or LoadImage's MASK (1=transparent, use mask_mode=opaque_is_0)")
        elif repainted == 0:
            report += " | nothing repainted (alpha is fully opaque)"
            print(f"[Pixel Grid] Restore Transparency: {report}")
            print("[Pixel Grid]   -> the mask says every pixel is solid; check "
                  "mask_mode (ComfyUI LoadImage MASK is 1=transparent)")
        else:
            print(f"[Pixel Grid] Restore Transparency: {report}")

        return (out, alpha, report)
