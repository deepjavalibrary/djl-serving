/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.serving;

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.wlm.WorkLoadManager;
import ai.djl.serving.wlm.WorkerPoolConfig;
import ai.djl.serving.workflow.BadWorkflowException;
import ai.djl.serving.workflow.Workflow;
import ai.djl.serving.workflow.WorkflowDefinition;
import ai.djl.util.Utils;

import org.apache.commons.cli.CommandLine;
import org.testng.Assert;
import org.testng.annotations.BeforeSuite;
import org.testng.annotations.Test;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;

public class WorkflowTest {

    /** A standard input for multiple tests with a zero image. */
    private Input zeroInput;

    public static void main(String[] args) throws BadWorkflowException, IOException {
        WorkflowTest t = new WorkflowTest();
        t.beforeAll();
        t.testCriteria();
    }

    @BeforeSuite
    public void beforeAll() throws IOException {
        ConfigManager.init(new Arguments(new CommandLine.Builder().build()));

        // Initialize zeroInput
        zeroInput = new Input();
        URL url = new URL("https://resources.djl.ai/images/0.png");
        try (InputStream is = url.openStream()) {
            zeroInput.add(Utils.toByteArray(is));
        }
    }

    @Test
    public void testJson() throws IOException, BadWorkflowException {
        Path workflowFile = Paths.get("src/test/resources/workflows/basic.json");
        runWorkflow(workflowFile, zeroInput);
    }

    @Test
    public void testYaml() throws IOException, BadWorkflowException {
        Path workflowFile = Paths.get("src/test/resources/workflows/basic.yaml");
        runWorkflow(workflowFile, zeroInput);
    }

    @Test
    public void testCriteria() throws IOException, BadWorkflowException {
        Path workflowFile = Paths.get("src/test/resources/workflows/criteria.json");
        runWorkflow(workflowFile, zeroInput);
    }

    @Test
    public void testFunctions() throws IOException, BadWorkflowException {
        Path workflowFile = Paths.get("src/test/resources/workflows/functions.json");
        runWorkflow(workflowFile, zeroInput);
    }

    @Test
    public void testLocalPerf() throws IOException, BadWorkflowException {
        Path workflowFile = Paths.get("src/test/resources/workflows/localPerf.json");
        Workflow workflow = WorkflowDefinition.parse(workflowFile).toWorkflow();
        WorkerPoolConfig<Input, Output> wpc = workflow.getWpcs().iterator().next();

        Assert.assertEquals(wpc.getQueueSize(), 102);
        Assert.assertEquals(wpc.getMaxIdleSeconds(), 62);
        Assert.assertEquals(wpc.getMaxBatchDelayMillis(), 302);
        Assert.assertEquals(wpc.getBatchSize(), 3);
    }

    @Test
    public void testEnsemble() throws IOException, BadWorkflowException {
        Path workflowFile = Paths.get("src/test/resources/workflows/ensemble.json");
        runWorkflow(workflowFile, zeroInput);
    }

    private Input runWorkflow(Path workflowFile, Input input)
            throws IOException, BadWorkflowException {
        Workflow workflow = WorkflowDefinition.parse(workflowFile).toWorkflow();
        WorkLoadManager wlm = new WorkLoadManager();
        for (WorkerPoolConfig<Input, Output> wpc : workflow.getWpcs()) {
            wpc.setMaxWorkers(1);
            wlm.registerWorkerPool(wpc).initWorkers("-1");
        }

        Output output = workflow.execute(wlm, input).join();
        workflow.close();
        wlm.close();
        Assert.assertNotNull(output.getData());
        return output;
    }
}
