from Sim.placer import *

import os
import importlib

# 2. Define the path to the plugins directory
CURRENT_DIR = os.path.dirname(__file__)
PLUGINS_DIR = os.path.join(CURRENT_DIR, '')
PACKAGE_NAME = __name__ # 'my_package'

def _load_subclasses():
    """Dynamically loads modules in the plugins directory."""
    
    # We load modules relative to the current package
    for filename in os.listdir(PLUGINS_DIR):
        if filename.endswith('.py') and filename != '__init__.py':
            # Example: 'my_package.plugins.sub_c'
            module_name = f"{PACKAGE_NAME}.{filename[:-3]}"
            try:
                # This executes the code in the subclass file, registering the class
                importlib.import_module(module_name)
            except ImportError as e:
                # Log or handle the error
                print(f"Warning: Could not load plugin {module_name}: {e}")

# 3. Execute the function immediately when the package is imported
_load_subclasses()