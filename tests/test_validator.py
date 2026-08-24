from contextlib import redirect_stdout
from io import StringIO
import unittest

from ontologybench import validate_outputs


class ValidatorTests(unittest.TestCase):
    def test_repeated_validation_is_deterministic(self):
        first = StringIO()
        with redirect_stdout(first):
            validate_outputs.main()

        second = StringIO()
        with redirect_stdout(second):
            validate_outputs.main()

        self.assertEqual(first.getvalue(), second.getvalue())

    def test_examples_do_not_inflate_issue_count(self):
        with redirect_stdout(StringIO()):
            validate_outputs.main()

        self.assertEqual(sum(validate_outputs.QC["warnings"].values()), 2)


if __name__ == "__main__":
    unittest.main()
