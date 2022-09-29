/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.huggingface.tokenizers;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.huggingface.translator.FillMaskTranslatorFactory;
import ai.djl.huggingface.translator.QuestionAnsweringTranslatorFactory;
import ai.djl.huggingface.translator.TextClassificationTranslatorFactory;
import ai.djl.huggingface.translator.TextEmbeddingTranslator;
import ai.djl.huggingface.translator.TextEmbeddingTranslatorFactory;
import ai.djl.huggingface.translator.TokenClassificationTranslatorFactory;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.modality.nlp.qa.QAInput;
import ai.djl.modality.nlp.translator.NamedEntity;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.testing.Assertions;
import ai.djl.translate.TranslateException;
import ai.djl.util.JsonUtils;
import ai.djl.util.Utils;

import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.Test;

import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class TranslatorTest {

    @AfterClass
    public void tierDown() {
        Utils.deleteQuietly(Paths.get("build/model"));
    }

    @Test
    public void testQATranslator() throws ModelException, IOException, TranslateException {
        String question = "When did BBC Japan start broadcasting?";
        String paragraph =
                "BBC Japan was a general entertainment Channel. "
                        + "Which operated between December 2004 and April 2006. "
                        + "It ceased operations after its Japanese distributor folded.";

        Block block =
                new LambdaBlock(
                        a -> {
                            NDManager manager = a.getManager();
                            long[][] start = new long[1][36];
                            long[][] end = new long[1][36];
                            start[0][0] = 2;
                            start[0][21] = 1;
                            end[0][0] = 2;
                            end[0][20] = 1;
                            NDArray arr1 = manager.create(start);
                            NDArray arr2 = manager.create(end);
                            return new NDList(arr1, arr2);
                        },
                        "model");
        Path modelDir = Paths.get("build/model");
        Files.createDirectories(modelDir);

        Criteria<QAInput, String> criteria =
                Criteria.builder()
                        .setTypes(QAInput.class, String.class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("tokenizer", "bert-base-cased")
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new QuestionAnsweringTranslatorFactory())
                        .build();

        try (ZooModel<QAInput, String> model = criteria.loadModel();
                Predictor<QAInput, String> predictor = model.newPredictor()) {
            QAInput input = new QAInput(question, paragraph);
            String res = predictor.predict(input);
            Assert.assertEquals(res, "December 2004");
        }

        Criteria<Input, Output> criteria2 =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("tokenizer", "bert-base-cased")
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new QuestionAnsweringTranslatorFactory())
                        .build();

        try (ZooModel<Input, Output> model = criteria2.loadModel();
                Predictor<Input, Output> predictor = model.newPredictor()) {
            Input input = new Input();
            input.add("question", question);
            input.add("paragraph", paragraph);
            Output res = predictor.predict(input);
            Assert.assertEquals(res.getAsString(0), "December 2004");
        }

        try (Model model = Model.newInstance("test")) {
            model.setBlock(block);
            Map<String, String> options = new HashMap<>();
            options.put("hasParameter", "false");
            model.load(modelDir, "test", options);

            QuestionAnsweringTranslatorFactory factory = new QuestionAnsweringTranslatorFactory();
            Map<String, String> arguments = new HashMap<>();

            Assert.assertThrows(
                    TranslateException.class,
                    () -> factory.newInstance(String.class, Integer.class, model, arguments));

            arguments.put("tokenizer", "bert-base-cased");

            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () -> factory.newInstance(String.class, Integer.class, model, arguments));
        }
    }

    @Test
    public void testFillMaskTranslator() throws ModelException, IOException, TranslateException {
        String text = "Hello I'm a [MASK] model.";

        Block block =
                new LambdaBlock(
                        a -> {
                            NDManager manager = a.getManager();
                            float[][] logits = new float[10][4828];
                            logits[6][4827] = 5;
                            logits[6][2535] = 4;
                            logits[6][2047] = 3;
                            logits[6][3565] = 2;
                            logits[6][2986] = 1;
                            NDArray arr = manager.create(logits);
                            arr = arr.expandDims(0);
                            return new NDList(arr);
                        },
                        "model");
        Path modelDir = Paths.get("build/model");
        Files.createDirectories(modelDir);

        Criteria<String, Classifications> criteria =
                Criteria.builder()
                        .setTypes(String.class, Classifications.class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("tokenizer", "bert-base-uncased")
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new FillMaskTranslatorFactory())
                        .build();

        try (ZooModel<String, Classifications> model = criteria.loadModel();
                Predictor<String, Classifications> predictor = model.newPredictor()) {
            Classifications res = predictor.predict(text);
            Assert.assertEquals(res.best().getClassName(), "fashion");
            Assert.assertThrows(
                    TranslateException.class,
                    () -> predictor.predict("Hello I'm a invalid model."));
            Assert.assertThrows(
                    TranslateException.class,
                    () -> predictor.predict("I'm a [MASK] [MASK] model."));
        }

        Criteria<Input, Output> criteria2 =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("tokenizer", "bert-base-uncased")
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new FillMaskTranslatorFactory())
                        .build();

        try (ZooModel<Input, Output> model = criteria2.loadModel();
                Predictor<Input, Output> predictor = model.newPredictor()) {
            Input input = new Input();
            input.add(text);
            Output out = predictor.predict(input);
            Classifications res = (Classifications) out.getData();
            Assert.assertEquals(res.best().getClassName(), "fashion");
        }

        try (Model model = Model.newInstance("test")) {
            model.setBlock(block);
            Map<String, String> options = new HashMap<>();
            options.put("hasParameter", "false");
            model.load(modelDir, "test", options);

            FillMaskTranslatorFactory factory = new FillMaskTranslatorFactory();
            Map<String, String> arguments = new HashMap<>();

            Assert.assertThrows(
                    TranslateException.class,
                    () -> factory.newInstance(String.class, Integer.class, model, arguments));

            arguments.put("tokenizer", "bert-base-uncased");

            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () -> factory.newInstance(String.class, Integer.class, model, arguments));
        }
    }

    @Test
    public void testTokenClassificationTranslator()
            throws ModelException, IOException, TranslateException {
        String text = "My name is Wolfgang and I live in Berlin.";

        Block block =
                new LambdaBlock(
                        a -> {
                            NDManager manager = a.getManager();
                            float[][] logits = new float[12][9];
                            logits[4][3] = 1;
                            logits[9][7] = 1;
                            NDArray arr = manager.create(logits);
                            arr = arr.expandDims(0);
                            return new NDList(arr);
                        },
                        "model");
        Path modelDir = Paths.get("build/model");
        Files.createDirectories(modelDir);
        Path path = modelDir.resolve("config.json");
        Map<String, Map<String, String>> map = new HashMap<>();
        Map<String, String> id2label = new HashMap<>();
        id2label.put("0", "O");
        id2label.put("3", "B-PER");
        id2label.put("7", "B-LOC");
        map.put("id2label", id2label);
        try (Writer writer = Files.newBufferedWriter(path)) {
            writer.write(JsonUtils.GSON.toJson(map));
        }

        Criteria<String, NamedEntity[]> criteria =
                Criteria.builder()
                        .setTypes(String.class, NamedEntity[].class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("tokenizer", "bert-base-uncased")
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new TokenClassificationTranslatorFactory())
                        .build();

        try (ZooModel<String, NamedEntity[]> model = criteria.loadModel();
                Predictor<String, NamedEntity[]> predictor = model.newPredictor()) {
            NamedEntity[] res = predictor.predict(text);
            Assert.assertEquals(res[0].getEntity(), "B-PER");
            Assertions.assertAlmostEquals(res[0].getScore(), 0.2536117);
            Assert.assertEquals(res[0].getIndex(), 4);
            Assert.assertEquals(res[0].getWord(), "wolfgang");
            Assert.assertEquals(res[0].getStart(), 11);
            Assert.assertEquals(res[0].getEnd(), 19);
        }

        Criteria<Input, Output> criteria2 =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("tokenizer", "bert-base-uncased")
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new TokenClassificationTranslatorFactory())
                        .build();

        try (ZooModel<Input, Output> model = criteria2.loadModel();
                Predictor<Input, Output> predictor = model.newPredictor()) {
            Input input = new Input();
            input.add(text);
            Output out = predictor.predict(input);
            NamedEntity[] res = (NamedEntity[]) out.getData().getAsObject();
            Assert.assertEquals(res[0].getEntity(), "B-PER");
        }

        try (Model model = Model.newInstance("test")) {
            model.setBlock(block);
            Map<String, String> options = new HashMap<>();
            options.put("hasParameter", "false");
            model.load(modelDir, "test", options);

            TokenClassificationTranslatorFactory factory =
                    new TokenClassificationTranslatorFactory();
            Map<String, String> arguments = new HashMap<>();

            Assert.assertThrows(
                    TranslateException.class,
                    () -> factory.newInstance(String.class, Integer.class, model, arguments));

            arguments.put("tokenizer", "bert-base-uncased");

            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () -> factory.newInstance(String.class, Integer.class, model, arguments));
        }
    }

    @Test
    public void testTextEmbeddingTranslatorForHuggingfaceModel() throws ModelNotFoundException, MalformedModelException, IOException, TranslateException {
        Criteria<String, float[]> criteria =
                Criteria.builder()
                        .setTypes(String.class, float[].class)
                        .optModelPath(Paths.get("/Users/ylwu/models/all-MiniLM-L6-v2_huggingface"))
                        .optEngine("PyTorch")
                        .optTranslatorFactory(new TextEmbeddingTranslatorFactory())
                        .build();
        ZooModel<String , float[]> model = criteria.loadModel();
        String sentence = "today is sunny";
        Predictor<String, float[]> predictor = model.newPredictor();
        float[] predict = predictor.predict(sentence);
        System.out.println(Arrays.toString(predict));
    }

    @Test
    public void testTextEmbeddingTranslatorForHuggingfaceModel2() throws ModelNotFoundException, MalformedModelException, IOException, TranslateException {
        String modelId = "sentence-transformers/all-MiniLM-L12-v2";
        Criteria<String, float[]> criteria =
                Criteria.builder()
                        .setTypes(String.class, float[].class)
                        .optModelUrls("djl://ai.djl.huggingface.pytorch/" + modelId)
                        .optEngine("PyTorch")
                        .optTranslatorFactory(new TextEmbeddingTranslatorFactory())
//                        .optTranslator(new TextEmbeddingTranslator())
                        .build();
        ZooModel<String , float[]> model = criteria.loadModel();
        String sentence = "today is sunny";
        Predictor<String, float[]> predictor = model.newPredictor();
        float[] predict = predictor.predict(sentence);
        System.out.println(Arrays.toString(predict));
        //[-0.08619279, 0.02554063, 0.060701434, -0.015559049, 0.058370855, -1.0265295E-4, 0.09983314, -0.09604423, -0.0411695, 0.018147089, 0.027398538, 0.024270587, -0.027863042, 0.08436614, 0.07513622, -0.016917754, 0.02565501, -0.05867155, -0.022904888, 0.001124735, -0.09503339, -0.043138616, -0.082055725, 0.05447176, 0.017103838, -0.011137141, 0.03439343, 0.029722685, 0.03339232, -0.0072781234, 0.0020839744, 0.026228134, -0.015226524, -0.056868926, -0.112452134, 0.0053706495, 0.03578312, -0.105457306, 8.1996125E-4, 0.02721722, 0.0035862552, -0.104514524, -0.04159897, 0.099233314, -0.011345826, 0.01836427, 0.015349504, 0.03226058, 0.036937624, 0.010055604, -0.027615547, -0.006096179, -0.045513637, -0.08285713, 0.049653217, 0.080971405, -0.0013517007, 0.09441161, 0.05774786, 0.006728853, 0.021342237, -0.007094511, 0.03689161, 0.01270418, 0.029900769, 0.0108382525, -0.051592525, -0.045669366, 0.01804641, -0.09505326, -0.06519111, -0.0051513417, -0.0482422, -0.020929603, -0.13400123, -0.060967542, -0.0040546404, -0.07165083, -0.057541884, -0.042066727, 0.03208687, -0.077733845, 0.023149299, 0.008523687, 0.014380634, 0.0676856, -0.044483602, 0.086830415, 0.024264304, -0.06096414, -0.056165356, 0.021171737, -0.056110464, -0.011212322, 0.024194652, 0.09260928, 0.006453035, 0.0294154, -0.024368946, 0.16097851, -0.03556941, -0.009298901, 0.0012660959, -0.03199064, 0.020361982, -0.037982713, 0.00430157, 0.0338242, -0.008357088, -0.058546606, -0.02504185, 0.06944696, 0.014827863, -0.012663232, 0.026298467, -0.01934025, 0.029811831, 0.06654935, -0.11477415, 0.023537345, -0.05974166, 0.040877804, 0.001450173, 0.026731666, 0.049120463, -0.06800859, -0.028994862, 0.017501418, 0.03023655, -0.0066655823, 0.02536841, -0.011845818, 0.08414076, -0.028045462, 0.018589586, 0.0446428, -0.062396184, -0.07685266, 0.06567167, -0.062177356, -0.040060107, 0.059693012, -0.047886215, 0.008409902, 0.11247223, 0.034929242, -0.05447774, 0.014907787, -0.042439416, 0.09704408, 0.06860466, -0.04079844, -0.038631484, -0.008182423, 0.043386042, 0.003200715, -0.032160085, 0.03367293, 0.12960123, -0.043676432, 0.081187926, 0.1019677, 0.043069065, -0.0787224, -0.079399206, -0.04243489, 0.025461733, 0.044015363, 0.0035656774, -0.015307716, 0.036827862, 0.0025080638, -0.021042809, 0.08137345, 0.044798553, -0.05015045, 0.011198546, 0.026348254, -0.033922367, -0.030194016, 0.02347479, 0.017305994, -0.03923834, 0.0022898964, 0.031042032, 9.306725E-4, -0.08654778, 0.08655228, -0.011375153, 0.027973106, -0.008816919, -0.05869056, -0.02322735, 0.080430895, -0.04242906, 0.06783263, 0.0023026217, -0.008480392, 0.0658785, 0.03397693, 0.023744835, -0.06554987, 0.021745129, 0.03912737, 0.11554025, -1.3449081E-5, 0.06977139, -0.009253652, -0.07487794, 0.030819746, -0.022439871, -0.0013604616, -0.0033675707, 0.107982464, -0.0595528, 0.07021031, -0.0420164, -0.023198403, -0.07028482, -0.007901463, 0.034215637, -0.05908873, 0.012916857, 1.311852E-32, 0.08202238, 0.012406462, -0.13925241, -0.041271325, -0.02275737, 0.010657995, 0.030038523, 0.06883527, -0.06507596, 0.008841926, -0.048438977, 0.025491616, -0.07533594, -0.038895074, -0.027221614, 0.040112067, -0.022664132, 0.025220038, -0.053116083, 0.01693213, -0.0430614, 9.0323377E-4, -0.041249525, 0.07489798, -0.028833715, 0.021363784, 0.034315627, 0.0018168573, -0.021215264, -0.025758713, -0.039467074, 0.055759028, -0.06703647, 0.07795877, -0.018991517, 0.06596877, 0.06493958, -0.115755394, -0.08551476, 0.040854856, -0.009791136, -0.07277594, 0.0629479, 0.03886115, -0.00907152, -0.06487296, -0.050792135, 0.054241322, -0.083845705, 0.047641966, -0.05531951, -0.10361827, 0.022450149, 0.054150593, 0.11353809, -0.0113452915, -0.10351095, -0.04782238, -0.02796283, 0.074173234, -0.0058270893, 0.013717544, 0.007654175, 0.052378584, 5.1411185E-5, 0.042133316, -0.008220739, 0.06373623, 0.03382809, 0.06694079, -0.011774219, -0.043672245, -0.06963908, 0.009118108, 0.04452668, 0.09658342, 0.037779003, 0.024519568, 0.038391784, 0.08372197, -0.053215638, -0.018670708, 0.014998552, -0.009968722, -0.022513473, 0.0074024894, 0.055557568, -0.06352399, -0.058395453, 0.014846777, -0.041776348, 0.06879345, -0.10525624, -0.026990954, -0.014635721, -5.956897E-33, 0.01876337, -0.075695716, -0.06410745, 0.008929181, 0.020960446, 0.09635463, 0.0028836755, -0.033280537, -0.020233728, -0.007587662, 0.055196874, -0.027418174, 0.09588196, 0.020585636, 0.024114784, -0.009816789, 0.014866806, -0.02161421, 0.022815809, 0.086427204, 0.052208383, 0.023224693, -0.038285345, 0.018823411, 0.09361607, 0.020613829, -0.03918713, 0.015373326, 7.0826296E-4, 0.08260815, 0.013425578, 0.03749219, 0.024105046, -0.07025552, -0.034095865, 0.0044403872, 0.024502568, 0.072068386, -0.013648618, 0.020518234, 0.042790473, 0.056797344, 0.0034029193, 0.00775409, 0.06511669, -0.09361175, -0.019058466, 0.0013460849, -0.04100223, -0.0592306, 0.028281312, -0.050615694, 0.06809166, 0.017115427, -0.024450338, -0.050986238, -0.025354827, -0.021847995, 0.04003914, -0.021586053, 0.009954797, -0.009233918, -0.08832905, 0.06446744]
        //[-0.08619279, 0.02554063, 0.060701434, -0.015559049, 0.058370855, -1.0265295E-4, 0.09983314, -0.09604423, -0.0411695, 0.018147089, 0.027398538, 0.024270587, -0.027863042, 0.08436614, 0.07513622, -0.016917754, 0.02565501, -0.05867155, -0.022904888, 0.001124735, -0.09503339, -0.043138616, -0.082055725, 0.05447176, 0.017103838, -0.011137141, 0.03439343, 0.029722685, 0.03339232, -0.0072781234, 0.0020839744, 0.026228134, -0.015226524, -0.056868926, -0.112452134, 0.0053706495, 0.03578312, -0.105457306, 8.1996125E-4, 0.02721722, 0.0035862552, -0.104514524, -0.04159897, 0.099233314, -0.011345826, 0.01836427, 0.015349504, 0.03226058, 0.036937624, 0.010055604, -0.027615547, -0.006096179, -0.045513637, -0.08285713, 0.049653217, 0.080971405, -0.0013517007, 0.09441161, 0.05774786, 0.006728853, 0.021342237, -0.007094511, 0.03689161, 0.01270418, 0.029900769, 0.0108382525, -0.051592525, -0.045669366, 0.01804641, -0.09505326, -0.06519111, -0.0051513417, -0.0482422, -0.020929603, -0.13400123, -0.060967542, -0.0040546404, -0.07165083, -0.057541884, -0.042066727, 0.03208687, -0.077733845, 0.023149299, 0.008523687, 0.014380634, 0.0676856, -0.044483602, 0.086830415, 0.024264304, -0.06096414, -0.056165356, 0.021171737, -0.056110464, -0.011212322, 0.024194652, 0.09260928, 0.006453035, 0.0294154, -0.024368946, 0.16097851, -0.03556941, -0.009298901, 0.0012660959, -0.03199064, 0.020361982, -0.037982713, 0.00430157, 0.0338242, -0.008357088, -0.058546606, -0.02504185, 0.06944696, 0.014827863, -0.012663232, 0.026298467, -0.01934025, 0.029811831, 0.06654935, -0.11477415, 0.023537345, -0.05974166, 0.040877804, 0.001450173, 0.026731666, 0.049120463, -0.06800859, -0.028994862, 0.017501418, 0.03023655, -0.0066655823, 0.02536841, -0.011845818, 0.08414076, -0.028045462, 0.018589586, 0.0446428, -0.062396184, -0.07685266, 0.06567167, -0.062177356, -0.040060107, 0.059693012, -0.047886215, 0.008409902, 0.11247223, 0.034929242, -0.05447774, 0.014907787, -0.042439416, 0.09704408, 0.06860466, -0.04079844, -0.038631484, -0.008182423, 0.043386042, 0.003200715, -0.032160085, 0.03367293, 0.12960123, -0.043676432, 0.081187926, 0.1019677, 0.043069065, -0.0787224, -0.079399206, -0.04243489, 0.025461733, 0.044015363, 0.0035656774, -0.015307716, 0.036827862, 0.0025080638, -0.021042809, 0.08137345, 0.044798553, -0.05015045, 0.011198546, 0.026348254, -0.033922367, -0.030194016, 0.02347479, 0.017305994, -0.03923834, 0.0022898964, 0.031042032, 9.306725E-4, -0.08654778, 0.08655228, -0.011375153, 0.027973106, -0.008816919, -0.05869056, -0.02322735, 0.080430895, -0.04242906, 0.06783263, 0.0023026217, -0.008480392, 0.0658785, 0.03397693, 0.023744835, -0.06554987, 0.021745129, 0.03912737, 0.11554025, -1.3449081E-5, 0.06977139, -0.009253652, -0.07487794, 0.030819746, -0.022439871, -0.0013604616, -0.0033675707, 0.107982464, -0.0595528, 0.07021031, -0.0420164, -0.023198403, -0.07028482, -0.007901463, 0.034215637, -0.05908873, 0.012916857, 1.311852E-32, 0.08202238, 0.012406462, -0.13925241, -0.041271325, -0.02275737, 0.010657995, 0.030038523, 0.06883527, -0.06507596, 0.008841926, -0.048438977, 0.025491616, -0.07533594, -0.038895074, -0.027221614, 0.040112067, -0.022664132, 0.025220038, -0.053116083, 0.01693213, -0.0430614, 9.0323377E-4, -0.041249525, 0.07489798, -0.028833715, 0.021363784, 0.034315627, 0.0018168573, -0.021215264, -0.025758713, -0.039467074, 0.055759028, -0.06703647, 0.07795877, -0.018991517, 0.06596877, 0.06493958, -0.115755394, -0.08551476, 0.040854856, -0.009791136, -0.07277594, 0.0629479, 0.03886115, -0.00907152, -0.06487296, -0.050792135, 0.054241322, -0.083845705, 0.047641966, -0.05531951, -0.10361827, 0.022450149, 0.054150593, 0.11353809, -0.0113452915, -0.10351095, -0.04782238, -0.02796283, 0.074173234, -0.0058270893, 0.013717544, 0.007654175, 0.052378584, 5.1411185E-5, 0.042133316, -0.008220739, 0.06373623, 0.03382809, 0.06694079, -0.011774219, -0.043672245, -0.06963908, 0.009118108, 0.04452668, 0.09658342, 0.037779003, 0.024519568, 0.038391784, 0.08372197, -0.053215638, -0.018670708, 0.014998552, -0.009968722, -0.022513473, 0.0074024894, 0.055557568, -0.06352399, -0.058395453, 0.014846777, -0.041776348, 0.06879345, -0.10525624, -0.026990954, -0.014635721, -5.956897E-33, 0.01876337, -0.075695716, -0.06410745, 0.008929181, 0.020960446, 0.09635463, 0.0028836755, -0.033280537, -0.020233728, -0.007587662, 0.055196874, -0.027418174, 0.09588196, 0.020585636, 0.024114784, -0.009816789, 0.014866806, -0.02161421, 0.022815809, 0.086427204, 0.052208383, 0.023224693, -0.038285345, 0.018823411, 0.09361607, 0.020613829, -0.03918713, 0.015373326, 7.0826296E-4, 0.08260815, 0.013425578, 0.03749219, 0.024105046, -0.07025552, -0.034095865, 0.0044403872, 0.024502568, 0.072068386, -0.013648618, 0.020518234, 0.042790473, 0.056797344, 0.0034029193, 0.00775409, 0.06511669, -0.09361175, -0.019058466, 0.0013460849, -0.04100223, -0.0592306, 0.028281312, -0.050615694, 0.06809166, 0.017115427, -0.024450338, -0.050986238, -0.025354827, -0.021847995, 0.04003914, -0.021586053, 0.009954797, -0.009233918, -0.08832905, 0.06446744]
        /*
        /usr/local/bin/python3.9 /Users/ylwu/code/os/ml-commons-test/nlp/test_sentence_transformer.py
[[-8.61928165e-02  2.55406350e-02  6.07014596e-02 -1.55590428e-02
   5.83708547e-02 -1.02660437e-04  9.98330861e-02 -9.60442498e-02
  -4.11695465e-02  1.81470700e-02  2.73984987e-02  2.42705997e-02
  -2.78630368e-02  8.43661800e-02  7.51362145e-02 -1.69177521e-02
   2.56549753e-02 -5.86715415e-02 -2.29048878e-02  1.12475106e-03
  -9.50334147e-02 -4.31386158e-02 -8.20557624e-02  5.44717833e-02
   1.71038117e-02 -1.11371689e-02  3.43934186e-02  2.97226626e-02
   3.33923250e-02 -7.27806287e-03  2.08397908e-03  2.62281727e-02
  -1.52265178e-02 -5.68689145e-02 -1.12452134e-01  5.37059689e-03
   3.57831120e-02 -1.05457284e-01  8.19970388e-04  2.72172503e-02
   3.58624174e-03 -1.04514502e-01 -4.15989794e-02  9.92333144e-02
  -1.13458205e-02  1.83642823e-02  1.53494673e-02  3.22606154e-02
   3.69376093e-02  1.00555932e-02 -2.76155025e-02 -6.09617773e-03
  -4.55136932e-02 -8.28571618e-02  4.96532209e-02  8.09714422e-02
  -1.35168771e-03  9.44116414e-02  5.77478521e-02  6.72886381e-03
   2.13422496e-02 -7.09452154e-03  3.68916020e-02  1.27041796e-02
   2.99007799e-02  1.08382609e-02 -5.15925661e-02 -4.56693545e-02
   1.80464163e-02 -9.50532407e-02 -6.51911274e-02 -5.15132025e-03
  -4.82422002e-02 -2.09296439e-02 -1.34001255e-01 -6.09675497e-02
  -4.05462785e-03 -7.16508180e-02 -5.75418919e-02 -4.20667417e-02
   3.20869200e-02 -7.77337849e-02  2.31493041e-02  8.52371566e-03
   1.43806525e-02  6.76856562e-02 -4.44835834e-02  8.68304446e-02
   2.42643133e-02 -6.09641261e-02 -5.61653338e-02  2.11717542e-02
  -5.61104678e-02 -1.12123080e-02  2.41946485e-02  9.26093012e-02
   6.45305868e-03  2.94153802e-02 -2.43689772e-02  1.60978511e-01
  -3.55693996e-02 -9.29893646e-03  1.26612757e-03 -3.19906287e-02
   2.03619786e-02 -3.79827470e-02  4.30158759e-03  3.38242054e-02
  -8.35711416e-03 -5.85465878e-02 -2.50418596e-02  6.94469810e-02
   1.48278670e-02 -1.26632107e-02  2.62984578e-02 -1.93402581e-02
   2.98118554e-02  6.65493533e-02 -1.14774212e-01  2.35373229e-02
  -5.97416237e-02  4.08777818e-02  1.45014655e-03  2.67316364e-02
   4.91204709e-02 -6.80085868e-02 -2.89948676e-02  1.75014269e-02
   3.02365348e-02 -6.66558230e-03  2.53684036e-02 -1.18458057e-02
   8.41407627e-02 -2.80454811e-02  1.85896084e-02  4.46427986e-02
  -6.23961501e-02 -7.68526420e-02  6.56717047e-02 -6.21773638e-02
  -4.00600620e-02  5.96930161e-02 -4.78862040e-02  8.40986986e-03
   1.12472229e-01  3.49292010e-02 -5.44777028e-02  1.49077708e-02
  -4.24394421e-02  9.70440507e-02  6.86046332e-02 -4.07984629e-02
  -3.86314839e-02 -8.18239804e-03  4.33860458e-02  3.20069375e-03
  -3.21600661e-02  3.36729176e-02  1.29601195e-01 -4.36764359e-02
   8.11879486e-02  1.01967655e-01  4.30690683e-02 -7.87224174e-02
  -7.93991983e-02 -4.24349047e-02  2.54617482e-02  4.40153815e-02
   3.56569490e-03 -1.53077105e-02  3.68278623e-02  2.50805495e-03
  -2.10427921e-02  8.13734680e-02  4.47985530e-02 -5.01504466e-02
   1.11985216e-02  2.63482425e-02 -3.39223631e-02 -3.01939696e-02
   2.34747771e-02  1.73059888e-02 -3.92383076e-02  2.28992547e-03
   3.10420692e-02  9.30668495e-04 -8.65477547e-02  8.65522698e-02
  -1.13751171e-02  2.79730987e-02 -8.81696213e-03 -5.86905740e-02
  -2.32273210e-02  8.04308951e-02 -4.24290746e-02  6.78326786e-02
   2.30258214e-03 -8.48037377e-03  6.58785254e-02  3.39769013e-02
   2.37448439e-02 -6.55498803e-02  2.17451043e-02  3.91273722e-02
   1.15540251e-01 -1.35078681e-05  6.97713569e-02 -9.25366115e-03
  -7.48779178e-02  3.08197662e-02 -2.24398542e-02 -1.36052782e-03
  -3.36757605e-03  1.07982464e-01 -5.95527738e-02  7.02102780e-02
  -4.20163907e-02 -2.31984798e-02 -7.02847987e-02 -7.90149253e-03
   3.42157073e-02 -5.90887144e-02  1.29168816e-02  1.31185243e-32
   8.20223913e-02  1.24064982e-02 -1.39252409e-01 -4.12713587e-02
  -2.27573570e-02  1.06579429e-02  3.00384928e-02  6.88352734e-02
  -6.50760010e-02  8.84196814e-03 -4.84390222e-02  2.54916288e-02
  -7.53358975e-02 -3.88950519e-02 -2.72215437e-02  4.01120745e-02
  -2.26641223e-02  2.52200738e-02 -5.31161167e-02  1.69321448e-02
  -4.30613756e-02  9.03164502e-04 -4.12495285e-02  7.48979747e-02
  -2.88337301e-02  2.13637874e-02  3.43156829e-02  1.81687006e-03
  -2.12152395e-02 -2.57586893e-02 -3.94671112e-02  5.57589792e-02
  -6.70364648e-02  7.79587701e-02 -1.89915132e-02  6.59688115e-02
   6.49395734e-02 -1.15755379e-01 -8.55147913e-02  4.08548191e-02
  -9.79111250e-03 -7.27759302e-02  6.29479587e-02  3.88611294e-02
  -9.07147955e-03 -6.48729578e-02 -5.07921726e-02  5.42413220e-02
  -8.38456601e-02  4.76419106e-02 -5.53194769e-02 -1.03618309e-01
   2.24501416e-02  5.41506298e-02  1.13538094e-01 -1.13453036e-02
  -1.03510983e-01 -4.78223823e-02 -2.79628541e-02  7.41732344e-02
  -5.82708651e-03  1.37175340e-02  7.65413791e-03  5.23785651e-02
   5.13898085e-05  4.21332866e-02 -8.22075643e-03  6.37362152e-02
   3.38280983e-02  6.69407994e-02 -1.17741637e-02 -4.36722897e-02
  -6.96390569e-02  9.11809225e-03  4.45266627e-02  9.65833887e-02
   3.77789959e-02  2.45195683e-02  3.83917987e-02  8.37219730e-02
  -5.32156788e-02 -1.86706800e-02  1.49985319e-02 -9.96871479e-03
  -2.25134157e-02  7.40244612e-03  5.55575415e-02 -6.35240451e-02
  -5.83954304e-02  1.48467869e-02 -4.17763628e-02  6.87934384e-02
  -1.05256252e-01 -2.69909427e-02 -1.46357492e-02 -5.95689623e-33
   1.87633485e-02 -7.56957009e-02 -6.41074404e-02  8.92916601e-03
   2.09604613e-02  9.63546112e-02  2.88367993e-03 -3.32804844e-02
  -2.02336982e-02 -7.58768711e-03  5.51968962e-02 -2.74182353e-02
   9.58819985e-02  2.05856282e-02  2.41148043e-02 -9.81680211e-03
   1.48668084e-02 -2.16141660e-02  2.28157938e-02  8.64271671e-02
   5.22083789e-02  2.32246649e-02 -3.82853821e-02  1.88234188e-02
   9.36160311e-02  2.06138529e-02 -3.91871259e-02  1.53733194e-02
   7.08260050e-04  8.26081932e-02  1.34255914e-02  3.74922045e-02
   2.41050776e-02 -7.02555254e-02 -3.40958685e-02  4.44041425e-03
   2.45025381e-02  7.20684230e-02 -1.36486273e-02  2.05182601e-02
   4.27904837e-02  5.67973033e-02  3.40290787e-03  7.75409350e-03
   6.51167035e-02 -9.36117396e-02 -1.90584622e-02  1.34607952e-03
  -4.10021879e-02 -5.92305735e-02  2.82813478e-02 -5.06156683e-02
   6.80917129e-02  1.71154700e-02 -2.44502984e-02 -5.09862378e-02
  -2.53547896e-02 -2.18480341e-02  4.00391519e-02 -2.15860568e-02
   9.95476637e-03 -9.23392270e-03 -8.83290991e-02  6.44674525e-02]]

Process finished with exit code 0

        * */
    }

    @Test
    public void testTextEmbeddingTranslator()
            throws ModelException, IOException, TranslateException {
        String text = "This is an example sentence";

        Block block =
                new LambdaBlock(
                        a -> {
                            NDManager manager = a.getManager();
                            NDArray arr = manager.ones(new Shape(1, 7, 384));
                            arr.setName("last_hidden_state");
                            return new NDList(arr);
                        },
                        "model");
        Path modelDir = Paths.get("build/model");
        Files.createDirectories(modelDir);

        Criteria<String, float[]> criteria =
                Criteria.builder()
                        .setTypes(String.class, float[].class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("tokenizer", "bert-base-uncased")
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new TextEmbeddingTranslatorFactory())
                        .build();

        try (ZooModel<String, float[]> model = criteria.loadModel();
                Predictor<String, float[]> predictor = model.newPredictor()) {
            float[] res = predictor.predict(text);
            Assert.assertEquals(res.length, 384);
            Assertions.assertAlmostEquals(res[0], 0.05103);
        }

        Criteria<Input, Output> criteria2 =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("tokenizer", "bert-base-uncased")
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new TextEmbeddingTranslatorFactory())
                        .build();

        try (ZooModel<Input, Output> model = criteria2.loadModel();
                Predictor<Input, Output> predictor = model.newPredictor()) {
            Input input = new Input();
            input.add(text);
            Output out = predictor.predict(input);
            float[] res = JsonUtils.GSON.fromJson(out.getAsString(0), float[].class);
            Assert.assertEquals(res.length, 384);
            Assertions.assertAlmostEquals(res[0], 0.05103);
        }

        try (Model model = Model.newInstance("test")) {
            model.setBlock(block);
            Map<String, String> options = new HashMap<>();
            options.put("hasParameter", "false");
            model.load(modelDir, "test", options);

            TextEmbeddingTranslatorFactory factory = new TextEmbeddingTranslatorFactory();
            Map<String, String> arguments = new HashMap<>();

            Assert.assertThrows(
                    TranslateException.class,
                    () -> factory.newInstance(String.class, Integer.class, model, arguments));

            arguments.put("tokenizer", "bert-base-uncased");

            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () -> factory.newInstance(String.class, Integer.class, model, arguments));
        }
    }

    @Test
    public void testTextClassificationTranslator()
            throws ModelException, IOException, TranslateException {
        String text = "DJL is the best.";

        Block block =
                new LambdaBlock(
                        a -> {
                            NDManager manager = a.getManager();
                            float[] logits = new float[] {0.02f, 0.2f, 0.97f};
                            NDArray arr = manager.create(logits, new Shape(1, 3));
                            arr.setName("logits");
                            return new NDList(arr);
                        },
                        "model");
        Path modelDir = Paths.get("build/model");
        Files.createDirectories(modelDir);

        Path path = modelDir.resolve("config.json");
        Map<String, Map<String, String>> map = new HashMap<>();
        Map<String, String> id2label = new HashMap<>();
        id2label.put("0", "LABEL_0");
        id2label.put("1", "LABEL_1");
        id2label.put("2", "LABEL_2");
        map.put("id2label", id2label);
        try (Writer writer = Files.newBufferedWriter(path)) {
            writer.write(JsonUtils.GSON.toJson(map));
        }

        Criteria<String, Classifications> criteria =
                Criteria.builder()
                        .setTypes(String.class, Classifications.class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("tokenizer", "bert-base-uncased")
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new TextClassificationTranslatorFactory())
                        .build();

        try (ZooModel<String, Classifications> model = criteria.loadModel();
                Predictor<String, Classifications> predictor = model.newPredictor()) {
            Classifications res = predictor.predict(text);
            Assert.assertEquals(res.best().getClassName(), "LABEL_2");
        }

        Criteria<Input, Output> criteria2 =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("tokenizer", "bert-base-uncased")
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new TextClassificationTranslatorFactory())
                        .build();

        try (ZooModel<Input, Output> model = criteria2.loadModel();
                Predictor<Input, Output> predictor = model.newPredictor()) {
            Input input = new Input();
            input.add(text);
            Output out = predictor.predict(input);
            Classifications res = (Classifications) out.getData();
            Assert.assertEquals(res.best().getClassName(), "LABEL_2");
        }

        try (Model model = Model.newInstance("test")) {
            model.setBlock(block);
            Map<String, String> options = new HashMap<>();
            options.put("hasParameter", "false");
            model.load(modelDir, "test", options);

            TextClassificationTranslatorFactory factory = new TextClassificationTranslatorFactory();
            Map<String, String> arguments = new HashMap<>();

            Assert.assertThrows(
                    TranslateException.class,
                    () -> factory.newInstance(String.class, Integer.class, model, arguments));

            arguments.put("tokenizer", "bert-base-uncased");

            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () -> factory.newInstance(String.class, Integer.class, model, arguments));
        }
    }
}
