# z-lstm

## What is it?

`z-lstm` is a basic implementation to run predictions with LSTM in [Orquestra](http://orquestra.io/docs/) – a platform for performing computations on quantum computers developed by [Zapata Computing](https://www.zapatacomputing.com).

## Usage

In order to use `z-lstm` in your workflow, you need to add it as a resource:

```yaml
resources:
- name: z-lstm
  type: git
  parameters:
    url: "git@github.com:zapatacomputing/z-lstm.git"
    branch: "master"
```

and then import in a specific step:

```yaml
- - name: my-task
    template: template-1
    arguments:
      parameters:
      - param_1: 1
      - resources: [z-lstm]
```

Once that is done, you can:
- use any template from `templates/` directory
- use tasks which import resource in the python code.

### Submitting workflow jobs

To submit a workflow to Orquestra, first login:

```
qe login -e <email> -s <server>
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

## Development and contribution

Create an virtual environment:

```
virtualenv venv
source venv/bin/activate
```

Install dependencies:

```
make install
```

### Running tests

Tests are located in `src/python/lstm/*_test.py` and can be run with:

```
make test
```

