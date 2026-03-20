package com.unmesh.skinpredictor;

public class PredictionResponse {
    private String label;
    private double score;
    private double threshold;

    public PredictionResponse(){

    }
    public String getLabel(){
        return label;
    }
    public void setLabel(String label){
        this.label = label;
    }
    public double getScore(){
        return score;
    }
    public void setScore(double score){
        this.score = score;
    }
    public double getThreshold(){
        return threshold;
    }
    public void setThreshold(double threshold){
        this.threshold = threshold;
    }
}
