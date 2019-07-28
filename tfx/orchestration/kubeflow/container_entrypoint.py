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
"""Main entrypoint for containers with Kubeflow TFX component executors.

We cannot use the existing TFX container entrypoint for the following
reason:Say component A requires inputs from component B output O1 and
component C output O2. Now, the inputs to A is a serialized dictionary
contained O1 and O2. But we need Argo to combine O1 and O2 into the expected
dictionary of artifact/artifact_type types, which isn't possible. Hence, we
need each output from a component to be individual argo output parameters so
they can be passed into downstream components as input parameters via Argo.

TODO(ajaygopinathan): The input names below are hardcoded and can easily
diverge from the actual names and types expected by the underlying executors.
Look into how we can dynamically generate the required inputs.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import sys

import six
import tensorflow as tf

from ml_metadata.proto import metadata_store_pb2
from tfx.orchestration import component_launcher
from tfx.orchestration import data_types
from tfx.types import artifact_utils
from tfx.types import channel
from tfx.utils import import_utils


def main():
  # Log to the container's stdout so Kubeflow Pipelines UI can display logs to
  # the user.
  logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  logging.getLogger().setLevel(logging.INFO)

  parser = argparse.ArgumentParser()
  parser.add_argument('--pipeline_name', type=str, required=True)
  parser.add_argument('--pipeline_root', type=str, required=True)
  parser.add_argument('--kubeflow_metadata_config', type=str, required=True)
  parser.add_argument('--component_id', type=str, required=True)
  parser.add_argument('--component_name', type=str, required=True)
  parser.add_argument('--component_type', type=str, required=True)
  parser.add_argument('--name', type=str, required=True)
  parser.add_argument('--driver_class_path', type=str, required=True)
  parser.add_argument('--executor_class_path', type=str, required=True)
  parser.add_argument('--inputs', type=str, required=True)
  parser.add_argument('--outputs', type=str, required=True)
  parser.add_argument('--exec_properties', type=str, required=True)
  parser.add_argument('--enable_cache', type=bool, default=True)

  args = parser.parse_args()

  component_info = data_types.ComponentInfo(
      component_type=args.component_type, component_id=args.component_id)

  inputs = artifact_utils.parse_artifact_dict(args.inputs)
  input_dict = {}
  for name, input_list in inputs.items():
    if not input_list:
      raise RuntimeError(
          'Found empty list of artifacts for input named {}: {}'.format(
              name, input_list))
    type_name = input_list[0].type_name
    input_dict[name] = channel.Channel(
        type_name=type_name, artifacts=input_list)

  outputs = artifact_utils.parse_artifact_dict(args.outputs)
  output_dict = {}
  for name, output_list in outputs.items():
    if not output_list:
      raise RuntimeError(
          'Found empty list of artifacts for output named {}: {}'.format(
              name, output_list))
    type_name = output_list[0].type_name
    output_dict[name] = channel.Channel(
        type_name=type_name, artifacts=output_list)

  exec_properties = json.loads(args.exec_properties)

  driver_class = import_utils.import_class_by_path(args.driver_class_path)
  executor_class = import_utils.import_class_by_path(args.executor_class_path)

  kubeflow_metadata_config = json.loads(args.kubeflow_metadata_config)
  connection_config = metadata_store_pb2.ConnectionConfig()

  connection_config.mysql.host = os.getenv(
      kubeflow_metadata_config['mysql_db_service_host_env_var'])
  connection_config.mysql.port = int(
      os.getenv(kubeflow_metadata_config['mysql_db_service_port_env_var']))
  connection_config.mysql.database = kubeflow_metadata_config['mysql_db_name']
  connection_config.mysql.user = kubeflow_metadata_config['mysql_db_user']
  connection_config.mysql.password = kubeflow_metadata_config[
      'mysql_db_password']

  pipeline_info = data_types.PipelineInfo(
      pipeline_name=args.pipeline_name,
      pipeline_root=args.pipeline_root,
      run_id=os.environ['WORKFLOW_ID'],
  )

  driver_args = data_types.DriverArgs(enable_cache=args.enable_cache)

  raw_args = exec_properties.get('beam_pipeline_args', [])

  # Beam expects str types for it's pipeline args. Ensure unicode type is
  # converted to str if required.
  beam_pipeline_args = []
  for arg in raw_args:
    # In order to support both Py2 and Py3: Py3 doesn't have `unicode` type.
    if six.PY2 and isinstance(arg, unicode):
      arg = arg.encode('ascii', 'ignore')

    beam_pipeline_args.append(arg)

  module_dir = os.environ['TFX_SRC_DIR']
  setup_file = os.path.join(module_dir, 'setup.py')
  tf.logging.info('Using setup_file \'%s\' to capture TFX dependencies',
                  setup_file)
  beam_pipeline_args.append('--setup_file={}'.format(setup_file))

  additional_pipeline_args = {
      'beam_pipeline_args': beam_pipeline_args,
  }

  launcher = component_launcher.BaseComponentLauncher(
      component_info=component_info,
      driver_class=driver_class,
      executor_class=executor_class,
      input_dict=input_dict,
      output_dict=output_dict,
      exec_properties=exec_properties,
      pipeline_info=pipeline_info,
      driver_args=driver_args,
      metadata_connection_config=connection_config,
      additional_pipeline_args=additional_pipeline_args)

  launcher.launch()


if __name__ == '__main__':
  main()
