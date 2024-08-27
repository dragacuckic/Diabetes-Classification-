/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Main.java to edit this template
 */
package nmdiabetes;

import java.util.ArrayList;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.eval.classification.ConfusionMatrix;
import org.neuroph.exam.NeurophExam;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.data.norm.MaxNormalizer;
import org.neuroph.util.data.norm.Normalizer;

/**
 *
 * @author Korisnik
 */
public class NMDiabetes implements NeurophExam, LearningEventListener{
    int inputCount = 8;
    int outputCount = 1;
    DataSet trainSet;
    DataSet testSet;
    double[] learningRates={0.2, 0.3, 0.4};
    int numOfIt=0;
    int numOfTr=0;
    ArrayList<Training> trainings = new ArrayList<>();

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        NMDiabetes nmd=new NMDiabetes();
        nmd.run();
    }

    @Override
    public DataSet loadDataSet() {
        return DataSet.createFromFile("diabetes_data.csv", inputCount, outputCount, ",");
    }

    @Override
    public DataSet preprocessDataSet(DataSet ds) {
        Normalizer norm=new MaxNormalizer(ds);
        norm.normalize(ds);
        
        ds.shuffle();
        return ds;
    }

    @Override
    public DataSet[] trainTestSplit(DataSet ds) {
        return ds.split(0.6, 0.4);
    }

    @Override
    public MultiLayerPerceptron createNeuralNetwork() {
        return new MultiLayerPerceptron(inputCount, 20, 16, outputCount);
    }

    @Override
    public MultiLayerPerceptron trainNeuralNetwork(MultiLayerPerceptron mlp, DataSet ds) {
        
        for (double lr : learningRates) {
            MomentumBackpropagation mbp = (MomentumBackpropagation) mlp.getLearningRule();
            mbp.addListener(this);
            mbp.setLearningRate(lr);
            mbp.setMaxError(0.07);
            mbp.setMomentum(0.5);
            mbp.setMaxIterations(1000);
            
            mlp.learn(trainSet);
            numOfTr++;
            numOfIt+=mbp.getCurrentIteration();
            
            
            evaluate(mlp, ds);
            
        }
        System.out.println("Srednja vrednost iteracija je: "+(double)numOfIt/numOfTr);
        
        
        return mlp;
    }

    @Override
    public void evaluate(MultiLayerPerceptron mlp, DataSet ds) {
        double accuracy=0;
        String[] klase=new String[]{"c1", "c2"};
        ConfusionMatrix cm=new ConfusionMatrix(klase);
        
        for (DataSetRow row : ds) {
            mlp.setInput(row.getInput());
            mlp.calculate();
            //OBRATI PAZNJU
            int actual=(int) Math.round(row.getDesiredOutput()[0]);
            int predicred=(int) Math.round(mlp.getOutput()[0]);
            
            
            cm.incrementElement(actual, predicred);
        }
        
        accuracy+=(double)(cm.getTruePositive(0)+cm.getTrueNegative(0))/cm.getTotal();
        
        System.out.println(cm.toString());
        
        System.out.println("Moj acc je "+accuracy);
        
        Training t = new Training(accuracy, mlp);
        trainings.add(t);
    }

    @Override
    public void saveBestNetwork() {
        Training maxTr= trainings.get(0);
        for (Training training : trainings) {
            if(training.getAccuracy()>maxTr.getAccuracy()) {
                maxTr=training;
            }
            
        }
        
        maxTr.getMlp().save("nn.nnet");
        System.out.println("Model je serijalizovan");
    }

    @Override
    public void handleLearningEvent(LearningEvent event) {
        MomentumBackpropagation mbp = (MomentumBackpropagation) event.getSource();
        System.out.println("Iterations: "+mbp.getCurrentIteration()+ "Total error: "+mbp.getTotalNetworkError());
    }

    private void run() {
        DataSet ds = loadDataSet();
        ds=preprocessDataSet(ds);
        DataSet[] trainAndTest=trainTestSplit(ds);
        trainSet=trainAndTest[0];
        testSet=trainAndTest[1];
        
        MultiLayerPerceptron mlp = createNeuralNetwork();
        trainNeuralNetwork(mlp, ds);
        saveBestNetwork();
        
    }

    private int getMax(double[] desiredOutput) {
        int max=0;
        for(int i=0;i<desiredOutput.length;i++) {
            if(desiredOutput[i]>desiredOutput[max]) {
                max=i;
            }
        }
        return max;
    }
    
}
