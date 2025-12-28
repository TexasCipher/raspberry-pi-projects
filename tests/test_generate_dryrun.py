import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import generate


class DryRunTests(unittest.TestCase):
    def test_dry_run_default(self):
        buf = io.StringIO()
        args = ["--dry-run", "--prompt", "Hello"]
        with redirect_stdout(buf):
            rc = generate.main(args)
        output = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("Hello [DRY RUN]", output)

    def test_dry_run_multiple_and_file(self):
        # create a temporary prompt file
        with tempfile.NamedTemporaryFile("w+", delete=False) as tf:
            tf.write("File prompt")
            tf.flush()
            tfname = tf.name
        try:
            buf = io.StringIO()
            args = ["--dry-run", "--file", tfname, "-n", "3"]
            with redirect_stdout(buf):
                rc = generate.main(args)
            output = buf.getvalue()
            self.assertEqual(rc, 0)
            self.assertIn("File prompt [DRY RUN]", output)
            # check that three sequences are present
            self.assertIn("(seq 1)", output)
            self.assertIn("(seq 2)", output)
            self.assertIn("(seq 3)", output)
        finally:
            try:
                os.unlink(tfname)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
