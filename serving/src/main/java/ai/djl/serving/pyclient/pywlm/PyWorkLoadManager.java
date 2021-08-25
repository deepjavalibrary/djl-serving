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
package ai.djl.serving.pyclient.pywlm;

import ai.djl.serving.pyclient.PythonConnector;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingDeque;

/** This class is responsible for managing the workload of the python worker threads. */
public class PyWorkLoadManager {
    private LinkedBlockingDeque<PyJob> jobQueue;
    private List<PyWorkerThread> workers;
    private ExecutorService threadPool;

    /**
     * Constructs a {@code PyWorkLoadManager} instance.
     *
     * @param queueSize size of the queue
     */
    PyWorkLoadManager(int queueSize) {
        jobQueue = new LinkedBlockingDeque<>(queueSize);
        workers = Collections.synchronizedList(new ArrayList<>());
        threadPool = Executors.newCachedThreadPool();
    }

    /**
     * Adds a job to the job queue.
     *
     * @param job job to be added.
     * @return whether job is added or not
     */
    public boolean addJob(PyJob job) {
        return jobQueue.offer(job);
    }

    /**
     * Adds a thread to worker pool.
     *
     * @param connector python connector
     * @param pythonPath path of the python
     */
    public void addThread(PythonConnector connector, String pythonPath) {
        PyWorkerThread workerThread =
                PyWorkerThread.builder()
                        .setPythonConnector(connector)
                        .setJobQueue(jobQueue)
                        .setPythonPath(pythonPath)
                        .build();

        workers.add(workerThread);
        threadPool.submit(workerThread);
    }

    /**
     * Returns a job queue.
     *
     * @return job queue
     */
    public LinkedBlockingDeque<PyJob> getJobQueue() {
        return jobQueue;
    }

    /**
     * Returns the list of workers.
     *
     * @return workers
     */
    public List<PyWorkerThread> getWorkers() {
        return workers;
    }
}
