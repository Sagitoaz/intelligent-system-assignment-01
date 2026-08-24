CREATE CONSTRAINT assignment_entity_key IF NOT EXISTS
FOR (n:AssignmentEntity) REQUIRE n.key IS UNIQUE;

MERGE (system:AssignmentEntity:System {key: 'diabetes-system'})
SET system.name = 'Diabetes Prediction System',
    system.domain = 'diabetes_assignment_01',
    system.purpose = 'Educational binary classification demonstration';

MERGE (model:AssignmentEntity:Model {key: 'diabetes-random-forest'})
SET model.name = 'Random Forest Classifier',
    model.domain = 'diabetes_assignment_01',
    model.task = 'Classification',
    model.n_estimators = 100,
    model.max_depth = 6,
    model.random_state = 42;

MERGE (target:AssignmentEntity:Target {key: 'diabetes-outcome'})
SET target.name = 'Outcome',
    target.domain = 'diabetes_assignment_01',
    target.classes = '0 = Non-diabetic; 1 = Diabetic';

MERGE (dataset:AssignmentEntity:Dataset {key: 'pima-diabetes-assignment-dataset'})
SET dataset.name = 'Diabetes dataset',
    dataset.domain = 'diabetes_assignment_01',
    dataset.observations = 768,
    dataset.source = 'Kaggle (as documented in the assignment notebook)';

MERGE (experiment:AssignmentEntity:Experiment {key: 'diabetes-final-selection'})
SET experiment.name = 'Final model selection',
    experiment.domain = 'diabetes_assignment_01',
    experiment.method = 'Training-only 5-fold cross-validation',
    experiment.selection_metric = 'F1-score',
    experiment.mean_cv_f1 = 0.6462;

UNWIND [
  {key:'feature-pregnancies', name:'Pregnancies', importance:0.0766800600062703},
  {key:'feature-glucose', name:'Glucose', importance:0.40171561704467146},
  {key:'feature-blood-pressure', name:'BloodPressure', importance:0.06433174087838672},
  {key:'feature-bmi', name:'BMI', importance:0.19854372880815993},
  {key:'feature-dpf', name:'DiabetesPedigreeFunction', importance:0.1243720742547756},
  {key:'feature-age', name:'Age', importance:0.13435677900773604}
] AS row
MERGE (feature:AssignmentEntity:Feature {key: row.key})
SET feature.name = row.name,
    feature.domain = 'diabetes_assignment_01',
    feature.raw_input = true
WITH feature, row
MATCH (model:AssignmentEntity:Model {key:'diabetes-random-forest'})
MERGE (model)-[uses:USES_FEATURE]->(feature)
SET uses.importance = row.importance,
    uses.importance_type = 'Random Forest impurity-based';

UNWIND [
  {key:'step-median-imputation', name:'Median Imputation', order:1},
  {key:'step-standard-scaling', name:'Standard Scaling', order:2}
] AS row
MERGE (step:AssignmentEntity:PipelineStep {key: row.key})
SET step.name = row.name,
    step.domain = 'diabetes_assignment_01',
    step.order = row.order
WITH step, row
MATCH (system:AssignmentEntity:System {key:'diabetes-system'})
MERGE (system)-[has:HAS_PIPELINE_STEP]->(step)
SET has.order = row.order;

UNWIND [
  {key:'metric-accuracy', name:'Accuracy', value:0.7468},
  {key:'metric-precision', name:'Precision', value:0.6667},
  {key:'metric-recall', name:'Recall', value:0.5556},
  {key:'metric-f1', name:'F1-score', value:0.6061}
] AS row
MERGE (metric:AssignmentEntity:Metric {key: row.key})
SET metric.name = row.name,
    metric.domain = 'diabetes_assignment_01',
    metric.value = row.value,
    metric.evaluation = 'held-out test set'
WITH metric
MATCH (model:AssignmentEntity:Model {key:'diabetes-random-forest'})
MERGE (model)-[:EVALUATED_BY]->(metric);

MATCH (system:AssignmentEntity:System {key:'diabetes-system'}),
      (model:AssignmentEntity:Model {key:'diabetes-random-forest'}),
      (target:AssignmentEntity:Target {key:'diabetes-outcome'}),
      (dataset:AssignmentEntity:Dataset {key:'pima-diabetes-assignment-dataset'}),
      (experiment:AssignmentEntity:Experiment {key:'diabetes-final-selection'})
MERGE (system)-[:USES_MODEL]->(model)
MERGE (model)-[:PREDICTS]->(target)
MERGE (model)-[:TRAINED_ON]->(dataset)
MERGE (experiment)-[:SELECTED]->(model)
MERGE (experiment)-[:COMPARES_REPRESENTATION]->(system);
