import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.nd4j.common.io.ClassPathResource;

import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_highgui.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class Example {
    public static void main(String[] args) throws Exception {
        String imageFile = new ClassPathResource("image/friends.jpg").getFile().getAbsolutePath();
        String outputFile = new ClassPathResource("image").getFile().getAbsolutePath() + "/image-output.jpg";

        Mat image = imread(imageFile);
        MTCNN mtcnn = new MTCNN();

        FaceAnnotation[] faceAnnotations = mtcnn.detectFace(image);
        List<Mat> alignedFace = new ArrayList<>();
        AgeGenderPredictor predictor = new AgeGenderPredictor();
        if(faceAnnotations.length != 0){
            for (FaceAnnotation faceAnnotation : faceAnnotations) {

                alignedFace.add(Utils.faceAligner(image, faceAnnotation));
                FaceAnnotation.BoundingBox bbox = faceAnnotation.getBoundingBox();
                Point x1y1 = new Point(bbox.getX(), bbox.getY());
                Point x2y2 = new Point(bbox.getX()+bbox.getW(), bbox.getY()+bbox.getH());
                rectangle(image, x1y1, x2y2, new Scalar(0, 255, 0, 0));
                for (FaceAnnotation.Landmark lm : faceAnnotation.getLandmarks()) {
                    Point keyPoint = new Point(lm.getPosition().getX(),lm.getPosition().getY());
                    circle(image, keyPoint, 2, new Scalar(0, 255, 0, 0), -1, 0, 0);
                }
            }
        }

        imwrite(outputFile, image);

        imshow("Input Image", image);

        for(int i = 0; i<alignedFace.size();i++){
            imshow("aligned_"+ (i + 1), alignedFace.get(i));
        }

        System.out.println("Perform predictions: ");
        for (Mat face: alignedFace)
        {
//            Mat resizedFace = null;
            resize(face, face, new Size(64,64));
            Prediction prediction = predictor.predict(face);
            System.out.println(prediction);
        }
        Thread.sleep(2000);
//        Press "Esc" to close window
        if (waitKey(0) == 27) {
            destroyAllWindows();
        }
    }
}
