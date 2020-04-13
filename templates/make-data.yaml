# Every template YAML file must begin with a `spec`, without which your template won't compile.
spec:

  # The `templates` section is where you list one or more templates
  templates:

  # This is the name of the template, which is used to reference it in the workflow. This field is required.
  - name: welcome-to-orquestra

    # `generic-task` is the supertemplate that all templates (that don't contain a `steps` section) must inherit from
    parent: generic-task

    # This section is for the inputs needed to run the template. This section is required.
    inputs:

      # `parameters` represent initialization values for a template. 
      parameters:

      # The `command` parameter is required because that is what is run by `generic-task`.
      - name: command
        value: python3 main.py

      # This section creates a script called `main.py` containing the code below under `data`. It must be under the `app` directory in order for the command above to locate it.
      artifacts:
      - name: main-script
        path: /app/main.py
        raw:
          data: |
            from orquestra import welcome
            welcome()

    # This section is where output artifacts are listed. They must be listed here, or else they will get deleted when the template completes. They must be under the `app` directory in order to be saved.
    outputs:
      artifacts:
      - name: welcome
        path: /app/welcome.json
