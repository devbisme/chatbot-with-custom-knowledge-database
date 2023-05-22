# Querying Local Documents Using LLMs

This is a simple Python app for using LLMs to query the content of one or more local documents.
This app is a modification of the code found in this [article](https://beebom.com/how-train-ai-chatbot-custom-knowledge-base-chatgpt-api) to use the current `llama-index`
package.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required Python packages.:

```bash
pip install -r reqs.txt
```

## Usage

1. Place the documents you wish to query in the `docs` directory.
2. Run the app: `python app.py`
3. Follow the instructions to open a browser window and submit queries about the documents.

Note there will be a delay the first time you use the app as your documents are processed and stored as
embeddings in a vector database.
Thereafter, the app should start very quickly since the saved database will be used.

If you add or remove items in the `documents` directory, remove the contents of the `indexes` directory
and the vector database will be regenerated.


## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](LICENSE)