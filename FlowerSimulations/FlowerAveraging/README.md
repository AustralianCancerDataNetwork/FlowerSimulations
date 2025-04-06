# Flower Example

This is a simple flower example to aggregate average of different columns present in a csv at local clients.

### Installing Dependencies

Project dependencies (such as `torch` and `flwr`) are defined in `pyproject.toml`. You can install the dependencies by invoking `pip`:

```shell
# From a new python environment, run:
pip install .
```

Then, to verify that everything works correctly you can run the following command:

```shell
python3 -c "import flwr"
```

If you don't see any errors you're good to go!

______________________________________________________________________

## Run Federated Learning with Flower

Afterwards you are ready to start the Flower server as well as the clients. You can simply start the server in a terminal as follows:

```shell
python3 server.py
```

Now you are ready to start the Flower clients which will participate in the learning. To do so simply open three more terminal windows and run the following commands.

Start client 1 in the first terminal:

```shell
python3 client_1.py
```

Start client 2 in the second terminal:

```shell
python3 client_2.py
```

Start client 3 in the second terminal:

```shell
python3 client_3.py
```

