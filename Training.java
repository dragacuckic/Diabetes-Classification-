/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package nmdiabetes;

import org.neuroph.nnet.MultiLayerPerceptron;

/**
 *
 * @author Korisnik
 */
public class Training {
    private double accuracy;
    private MultiLayerPerceptron mlp;

    public Training(double accuracy, MultiLayerPerceptron mlp) {
        this.accuracy = accuracy;
        this.mlp = mlp;
    }

    public double getAccuracy() {
        return accuracy;
    }

    public void setAccuracy(double accuracy) {
        this.accuracy = accuracy;
    }

    public MultiLayerPerceptron getMlp() {
        return mlp;
    }

    public void setMlp(MultiLayerPerceptron mlp) {
        this.mlp = mlp;
    }

    
    

    
    
}
