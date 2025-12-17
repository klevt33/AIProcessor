## Adding a New Classifier
You must create these new things if you want to add a new classifier
* Update `job_config.yaml`
    * add the classifier name to the list of classifiers
* new YAML config file:
    * `(classifier name)_config.yaml`
    * look at other classifiers, will have additional classifier specific information
* Storage bucket
    * `spendreport-automated-(classifier name)-classifier`
    * this is where the training data will be stored for the classifier
* Upload all of the already "USED" training data to the new storage bucket
    1. In the `job_config.yaml` file, comment out all classifiers except for the new one
    2. Temporarily set `RELOAD_ALL_TRAINING_DATA` to `true`
    4. Run the pipeline to upload all of the training data to the new storage bucket
    5. Reset `RELOAD_ALL_TRAINING_DATA` to `false`
    6. Re-add all of the original classifiers in the `job_config.yaml` file
