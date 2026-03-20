package com.unmesh.skinpredictor;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;

@Controller
public class HomeController {
    @Autowired
    private PredictionService predictionService;

    @GetMapping("/")
    public String home(){
        return "index";
    }

    @PostMapping("/upload")
    public String uploadImage(@RequestParam("image") MultipartFile image, Model model){
        if(image.isEmpty()){
            model.addAttribute("message", "Please select an Image file");
            return "index";
        }
        try{
            PredictionResponse prediction = predictionService.getPrediction(image);
            model.addAttribute("message", "Image uploaded successfully.");
            model.addAttribute("fileName", image.getOriginalFilename());
            model.addAttribute("label", prediction.getLabel());
            model.addAttribute("score", prediction.getScore());
            model.addAttribute("threshold", prediction.getThreshold());

        } catch (Exception e){
            model.addAttribute("message", "Prediction failed : " + e.getMessage());
        }
        
        return "index";
    }
}
