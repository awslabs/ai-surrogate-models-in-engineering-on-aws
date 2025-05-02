.. _quickstart-cli-framework:

Creating Custom CLI Commands
============================

MLSimKit is designed for you to extend with your own commands and tools. ML4Simkit uses Click integrated 
with an opinionated configuration system based on Pydantic schemas. You can use this CLI framework 
to build your own commands and schema-based configs. Continue reading below for a quick guide. See the 
:ref:`API reference <api-cli>` for details. 

For example, this code creates a command that automatically exposed the ``DB`` fields as CLI options:

.. code-block:: python

    # connect.py
    import pydantic
    import click
    import json
    import mlsimkit

    class DB(pydantic.BaseModel):
        user: str
        password: str
        port: int = 5000
        host: str = "localhost"

    @click.command(cls=mlsimkit.cli.BaseCommand)
    @mlsimkit.cli.options(DB, dest='db')
    def connect(db: DB):
        print(json.dumps(db.dict(), indent=2))

    if __name__ == '__main__':
        connect()


.. code-block:: shell

    $ python3 connect.py --help
    Usage: connect.py [OPTIONS]

    Options:
      --user TEXT      [required]
      --password TEXT  [required]
      --port INTEGER
      --host TEXT
      --help           Show this message and exit.

    $ python3 connect.py --user user1 --password "***"
    {
      "user": "user1",
      "password": "***",
      "port": 5000,
      "host": "localhost"
    }

You can use commands on their own or combine them into a set of commands as part of a larger program.

A program consists of one or more commands and automates features such as loading YAML config tree.

**Example program**

.. code-block:: python

    import pydantic
    import click
    import json
    import mlsimkit

    class DB(pydantic.BaseModel):
        user: str
        password: str
        port: int = 5000
        host: str = "localhost"

    class QueryRequest(pydantic.BaseModel):
        table_name: str
        query: str

    @mlsimkit.cli.program(name="mytool", version="0.1", use_config_file=True, chain=True)
    def mytool(ctx: click.Context, config_file: click.Path):
        ctx.obj["config_file"] = config_file
        pass

    @mytool.command()
    @mlsimkit.cli.options(DB, dest='db', allow_yaml_file=True)
    def connect(ctx: click.Context, db: DB):
        print("Connect: ", json.dumps(db.dict(), indent=2))

    @mytool.command()
    @mlsimkit.cli.options(QueryRequest, dest='request')
    def query(ctx: click.Context, request: QueryRequest):
        print("Query: ", json.dumps(request.dict(), indent=2))

    if __name__ == '__main__':
        mytool()

**Program --help**

.. code-block:: shell

    % python3 mytool.py --help
    Usage: mytool.py [OPTIONS] COMMAND1 [ARGS]... [COMMAND2 [ARGS]...]...

    Options:
      --version      Show the version and exit.
      --config PATH  Path to YAML config file. NOTE: overrides command-line
                     arguments.
      --help         Show this message and exit.

    Commands:
      connect
      query

A **program config** can be used to set commands options automatically:

.. code-block:: yaml

    connect:
      db:
        user: "user1"
        password: "***"
        host: 127.0.0.1
        port: 8888

    query:
      request:
        table_name: "mytable"
        query: "?items=*"

A program **runs commands from config** automatically:

.. code-block:: shell

    % python3 mytool.py --config config.yaml connect
    Connect:  {
      "user": "user1",
      "password": "***",
      "port": 8888,
      "host": "127.0.0.1"
    }

    % python3 mytool.py --config mydb.yaml query
    Query:  {
      "table_name": "mytable",
      "query": "?items=*"
    }

You may **override the config with command-line** options:

.. code-block:: shell

    % python3 mytool.py --config config.yaml query --query "?items=user"
    Query:  {
      "table_name": "mytable",
      "query": "?items=user"
    }

You can use the config while **chaining commands together**:

.. code-block:: shell

    % python3 mytool.py --config config.yaml connect query --query "?items=user" query --query "?items=roles"
    Connect:  {
      "user": "user1",
      "password": "***",
      "port": 8888,
      "host": "127.0.0.1"
    }
    Query:  {
      "table_name": "mytable",
      "query": "?items=user"
    }
    Query:  {
      "table_name": "mytable",
      "query": "?items=roles"
    }

You can **reference YAML files** from a "defaults" section in a config:

.. code-block:: text

    config.yaml
    db
    ├── local.yaml
    └── remote.yaml

.. code-block:: yaml

    # config.yaml
    connect:
      defaults:
        db: +db/local

      db:
        user: admin

.. code-block:: shell

    # Connect to local by default (config.yaml)
    % python3 mytool.py --config config.yaml connect --password "****"
    Connect:  {
      "user": "admin",
      "password": "****",
      "port": 8888,
      "host": "127.0.0.1"
    }

    # Connect to remote
    % python3 mytool.py --config config.yaml connect --db db/remote --user admin --password "****"
    Connect:  {
      "user": "admin",
      "password": "****",
      "port": 5000,
      "host": "198.1.1.100"
    }


For further details, see the :ref:`CLI framework API documentation<api-cli>`
