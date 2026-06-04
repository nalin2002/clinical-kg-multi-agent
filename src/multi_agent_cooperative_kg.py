"""
Cooperative multi-agent clinical KG extraction pipeline.

This module is a thin CLI shim. Implementation lives in the ``cooperative_kg``
package so constants, clients, agents, and validation stay in focused modules.
"""

from cooperative_kg.cli import main

if __name__ == "__main__":
    main()
