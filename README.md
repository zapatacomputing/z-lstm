# z-lstm

This repository contain the source code and workflow templates to run [Prediction with LSTM](http://orquestra.io/docs/tutorial/lstm/) in Orquestra.

Example workflows can be found inside `example/`.

## Submiting workflow jobs

To submit a workflow to Orquestra, first login:

```
qe login -e steinkirch@zapatacomputing.com -s http://prod-b.orquestra.io
```

Then submit with:

```
qe submit workflow example/lstm-tutorial.yaml
```

You can check the workflow processing with:

```
qe get workflow <workflow_id>
```

You can check logs with:

```
qe get logs <workflow_id> -s <step_id>
```

Finally, you can get workflow results with:

```
qe get workflowresult  <workflow_id>
```


### Plotting results

To plot the results, run:

```
python examples/plot_lstm.py <workflow result JSON>
```


---

## Developing

Create an virtual environment:

```
virtualenv venv
source venv/bin/activate
```

Install depedencies:

```
make install
```

### Running tests

Tests are located in `src/python/lstm/test`. Run them with:

```
make test
```

