import org.bytedeco.opencv.opencv_core.Mat;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import java.io.File;
import java.io.IOException;


public class AgeGenderPredictor {

    private static final Logger log = LoggerFactory.getLogger(AgeGenderPredictor.class);
    private static String modelPath = "model\\age_gender_est.hdf5";
    private static ComputationGraph model;


    public AgeGenderPredictor() throws InvalidKerasConfigurationException, IOException, UnsupportedKerasConfigurationException {
        String file = new File("D:\\Repos\\rude-carnie-dl4j-model-import\\model\\age_gender_est.hdf5").getAbsolutePath();
        model = KerasModelImport.importKerasModelAndWeights(file,new int[]{64,64,3}, false);
        System.out.println("Model loaded successfully...");
        System.out.println(model.summary());
    }


    public Prediction predict(Mat image) throws IOException{
        NativeImageLoader loader = new NativeImageLoader();
        INDArray indarrayImg = loader.asMatrix(image);
        System.out.println("before reshape: " + indarrayImg.shapeInfoToString());
        INDArray channellast = indarrayImg.permute(0, 2, 3, 1);
        System.out.println("after reshape: " + channellast.shapeInfoToString());
        INDArray[] results = model.output(channellast);
        System.out.println(results.length);
        INDArray predictedGenders = results[0];
        INDArray ages = Nd4j.arange(0,101).reshape(new int[] {101,1});
        INDArray predicted_ages = results[1].mmul(ages).ravel();
        return new Prediction(predictedGenders, predicted_ages);
    }

}
