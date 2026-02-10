
import sys
from unittest.mock import MagicMock

def apply_mocks():
    """
    Mocks dependencies that might be missing in some environments
    to allow GATr and other models to run without full installation.
    """
    # Mock dependencies if they are not present
    for mod in ["opt_einsum", "xformers", "xformers.ops"]:
        try:
            __import__(mod)
        except ImportError:
            class MockType(type):
                pass
            class MockClass:
                pass
            
            mock_mod = MagicMock()
            if mod == "xformers.ops":
                mock_mod.AttentionBias = MockClass
            sys.modules[mod] = mock_mod
        
    print("Mocks applied successfully.")
