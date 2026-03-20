package com.unmesh.skinpredictor;

import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

@Service
public class PredictionService {
    private final String FLASK_API_URL = "http://127.0.0.1:5000/predict";
    public PredictionResponse getPrediction(MultipartFile image) throws Exception{
        RestTemplate restTemplate = new RestTemplate();
        ByteArrayResource fileAsResource = new ByteArrayResource(image.getBytes()){
            @Override
            public String getFilename(){
                return image.getOriginalFilename();
            }
        };
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", fileAsResource);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        ResponseEntity<PredictionResponse> response = restTemplate.exchange(FLASK_API_URL, HttpMethod.POST, requestEntity, PredictionResponse.class);

        return response.getBody();
    }

}
