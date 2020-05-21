import org.nd4j.linalg.api.ndarray.INDArray;

public class Prediction {

    private INDArray genderProbability;
    private INDArray ageArray;
    private String gender;
    private int age;

    public Prediction(INDArray genderArray, INDArray age)
    {
        this.genderProbability = genderArray;
        this.ageArray = age;

        if(genderProbability.getDouble(0) < 0.5)
        {
            gender = "male";
        }
        else
        {
            gender = "female";
        }

        this.age = (int) ageArray.getDouble(0);
    }

    public String getGender()
    {
        return gender;
    }

    public double getAge()
    {
        return age;
    }



    @Override
    public String toString() {
        return "Prediction{" +
                "genderProbability=" + genderProbability +
                ", ageArray=" + ageArray +
                ", gender='" + gender + '\'' +
                ", age=" + age +
                '}';
    }
}
