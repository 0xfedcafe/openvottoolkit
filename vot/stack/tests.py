"""Tests for the experiment stack module."""

import unittest
import yaml

from vot.workspace import NullStorage
from vot.stack import Stack, list_integrated_stacks, resolve_stack

class NoWorkspace:
    """Empty workspace, does not save anything."""

    @property
    def storage(self) -> NullStorage:
        """Returns the storage object for the workspace."""
        return NullStorage()

class TestStacks(unittest.TestCase):
    """Tests for the experiment stack utilities."""

    def test_stacks(self) -> None:
        """Test loading integrated stacks."""

        stacks = list_integrated_stacks()
        for stack_name in stacks:
            try:
                stack_path = resolve_stack(stack_name)
                if stack_path is None:
                    self.fail("Stack {} not found".format(stack_name))
                    continue
                with open(stack_path, 'r') as fp:
                    stack_metadata = yaml.load(fp, Loader=yaml.BaseLoader)
                    Stack(stack_name, NoWorkspace(), **stack_metadata)
            except Exception as e:
                self.fail("Stack {}: {}".format(stack_name, e))
