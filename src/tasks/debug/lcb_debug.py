import json
import pickle
import zlib
from dataclasses import dataclass
from enum import Enum

class TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"

@dataclass
class Test:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        self.testtype = TestType(self.testtype)

class CodeGenerationProblem:
    def __init__(self, data_dict: dict):
        self.question_content = data_dict["question_content"]
        self.starter_code = data_dict["starter_code"] if len(data_dict["starter_code"]) != 0 else None
        self.public_test_cases = json.loads(data_dict["public_test_cases"])
        self.public_test_cases = [Test(**t) for t in self.public_test_cases]
        self.metadata = json.loads(data_dict["metadata"]) if "metadata" in data_dict else {}
        
        try:
            self.private_test_cases = json.loads(data_dict["private_test_cases"])  # type: ignore
        except:
            self.private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(data_dict["private_test_cases"].encode("utf-8"))  # type: ignore
                    )
                )
            )  # type: ignore
        self.private_test_cases = [Test(**t) for t in self.private_test_cases]
        
    def get_evaluation_sample(self):
        return {
            "input_output": json.dumps(
                {
                    "inputs": [
                        t.input
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "outputs": [
                        t.output
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "fn_name": self.metadata.get("func_name", None),
                }
            ),
        }