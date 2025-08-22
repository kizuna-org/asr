from file_fetcher import FileFetcher


if __name__ == "__main__":
    root = "/mnt/samba"
    name = "aaa"
    version = "1"
    file = "file"

    fetcher = FileFetcher(root)
    fetcher.set_options(name=name, version=version, file=file)

    file_path = fetcher.get_file_path()
    print(f"File path: {file_path}")

    contents = fetcher.get_file_contents()
    if contents is not None:
        print("File contents:")
        print(contents)
    else:
        print("File not found or could not be read.")
