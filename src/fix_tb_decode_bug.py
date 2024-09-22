import string


def replace_invalid_utf8(input_file, output_file):
    printable_chars = set(string.printable)
    with open(input_file, "rb") as f_in, open(output_file, "wb") as f_out:
        while True:
            chunk = f_in.read(1024)
            if not chunk:
                break
            try:
                chunk_decoded = chunk.decode("utf-8")
            except UnicodeDecodeError:
                chunk_decoded = chunk.decode("utf-8", errors="replace").replace(
                    "�", "x"
                )
            chunk_decoded = "".join(
                c if c in printable_chars else "x" for c in chunk_decoded
            )
            f_out.write(chunk_decoded.encode("utf-8"))
