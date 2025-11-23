"""
Factory module for creating FlashAttention operator and hardware instances.

This module decouples the dashboard from specific operator implementations,
allowing other interfaces to easily instantiate the appropriate components.
"""

from typing import Any, Dict, Optional, Tuple

from hardware_descriptions import FlashAttentionHardware
from .operators.flash_attention_operator import FlashAttentionOperator
from .operators.llmcompass_flash_attention import (
    LLMCompassFlashAttentionOperator,
    make_llmcompass_hardware,
)

# Constants for implementation types
MANUAL_IMPL = "Manual configuration"
LLM_PREFIX = "LLMCompass:"


def get_implementation(
    impl_name: str,
    workload_metadata: Dict[str, Any],
    manual_hardware: FlashAttentionHardware,
    llm_devices: Dict[str, Any],
) -> Tuple[Optional[FlashAttentionOperator], Optional[FlashAttentionHardware], str]:
    """
    Factory to create operator and hardware instances based on implementation name.
    
    Args:
        impl_name: Name of the implementation (e.g., "Manual..." or "LLMCompass: ...").
        workload_metadata: Dictionary of workload parameters.
        manual_hardware: The manually configured hardware object.
        llm_devices: Dictionary of available LLMCompass devices.
        
    Returns:
        A tuple containing:
        - Operator instance (or None)
        - Hardware instance (or None)
        - Label string for display (or error message)
    """
    if impl_name == MANUAL_IMPL:
        return FlashAttentionOperator(workload_metadata), manual_hardware, MANUAL_IMPL
    
    if impl_name.startswith(LLM_PREFIX):
        device_name = impl_name.split(":", 1)[1].strip()
        device = llm_devices.get(device_name)
        if device is None:
            return None, None, f"Device '{device_name}' not found"
            
        hardware = make_llmcompass_hardware(device_name, device=device)
        operator = LLMCompassFlashAttentionOperator(workload_metadata)
        return operator, hardware, f"LLMCompass • {device_name}"
        
    return None, None, "Unknown implementation"
