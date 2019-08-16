# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Integration tests for Kubeflow-based orchestrator and GCP backend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys

import tensorflow as tf

from tfx.components.evaluator.component import Evaluator
from tfx.components.example_gen.csv_example_gen.component import CsvExampleGen
from tfx.components.model_validator.component import ModelValidator
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components.transform.component import Transform
from tfx.orchestration.kubeflow import test_utils
from tfx.proto import evaluator_pb2
from tfx.utils import dsl_utils


class KubeflowGCPIntegrationTest(test_utils.BaseKubeflowTest):

  def testDataflowRunner(self):
    """Test for DataflowRunner invocation."""
    pipeline_name = 'kubeflow-dataflow-test-{}'.format(self._random_id())

    pipeline = self._create_dataflow_pipeline(pipeline_name, [
        CsvExampleGen(input_base=dsl_utils.csv_input(self._data_root)),
        StatisticsGen(input_data=self._mock_raw_examples),
        Transform(
            input_data=self._mock_raw_examples,
            schema=self._mock_schema,
            module_file=self._taxi_module_file),
        Evaluator(
            examples=self._mock_raw_examples,
            model_exports=self._mock_model,
            feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(specs=[
                evaluator_pb2.SingleSlicingSpec(
                    column_for_slicing=['trip_start_hour'])
            ])),
        ModelValidator(
            examples=self._mock_raw_examples, model=self._mock_model),
    ])

    self._compile_and_run_pipeline(pipeline)

  # TODO(muchida): Add test cases for AI Platform Trainer and Pusher.


if __name__ == '__main__':
  logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  tf.test.main()
