import argparse
import dataclasses
from dataclasses import dataclass

@dataclass
class AsyncEngineArgs:
    
    @staticmethod
    def add_cli_args(
            parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser
    
    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'AsyncEngineArgs':
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return engine_args
        