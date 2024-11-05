package com.example.voicebot;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.View;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Locale;
import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "VoiceBot";
    private TextToSpeech textToSpeech;
    private SpeechRecognizer speechRecognizer;
    private TextView responseText;
    private ImageButton micButton;
    private OkHttpClient client;
    private static final String LLM_API_URL = "https://api-inference.huggingface.co/models/__modelname__";
    private static final String LLM_API_KEY = "API_KEY"; // Replace with your actual OpenAI API key

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        responseText = findViewById(R.id.responseText);
        micButton = findViewById(R.id.micButton);
        client = new OkHttpClient();

        // Initialize TextToSpeech
        textToSpeech = new TextToSpeech(this, status -> {
            if (status == TextToSpeech.SUCCESS) {
                textToSpeech.setLanguage(Locale.US);
            } else {
                Log.e(TAG, "TextToSpeech initialization failed");
            }
        });

        // Initialize SpeechRecognizer
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this);
        speechRecognizer.setRecognitionListener(new RecognitionListener() {
            @Override
            public void onResults(Bundle results) {
                ArrayList<String> matches = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
                if (matches != null && !matches.isEmpty()) {
                    String userQuery = matches.get(0); // This is where userQuery is obtained
                    Log.d(TAG, "User Query: " + userQuery);
                    getResponseFromLLM(userQuery); // userQuery is passed to the method here
                } else {
                    Log.e(TAG, "No speech matches found.");
                    Toast.makeText(MainActivity.this, "No speech recognized", Toast.LENGTH_SHORT).show();
                }
            }


            @Override public void onReadyForSpeech(Bundle params) {}
            @Override public void onBeginningOfSpeech() {}
            @Override public void onRmsChanged(float rmsdB) {}
            @Override public void onBufferReceived(byte[] buffer) {}
            @Override public void onEndOfSpeech() {}
            @Override public void onError(int error) {
                Log.e(TAG, "Speech recognition error: " + error);
                Toast.makeText(MainActivity.this, "Error recognizing speech", Toast.LENGTH_SHORT).show();
            }
            @Override public void onPartialResults(Bundle partialResults) {}
            @Override public void onEvent(int eventType, Bundle params) {}
        });

        micButton.setOnClickListener(v -> startListening());

        // Request microphone permission
        requestAudioPermission();
    }

    private void startListening() {
        Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault());
        try {
            speechRecognizer.startListening(intent);
        } catch (Exception e) {
            Log.e(TAG, "Error starting speech recognizer: " + e.getMessage());
            Toast.makeText(this, "Error starting speech recognition", Toast.LENGTH_SHORT).show();
        }
    }

    private void getResponseFromLLM(String userQuery) {
        responseText.setText("Processing your question...");

        // Prepare JSON body for the OpenAI API request
        MediaType JSON = MediaType.parse("application/json; charset=utf-8");
        JSONObject jsonBody = new JSONObject();
        try {
            jsonBody.put("model", "EleutherAI/gpt-neo-2.7B");
            jsonBody.put("prompt", userQuery);
            jsonBody.put("max_tokens", 100);
            jsonBody.put("temperature", 0.7); // Optional: adjust as needed
            jsonBody.put("top_p", 1.0);       // Optional: adjust as needed
            jsonBody.put("n", 1);              // Optional: adjust as needed
            jsonBody.put("stop", JSONObject.NULL); // Optional: specify stopping sequences if needed

        } catch (Exception e) {
            Log.e(TAG, "Error creating JSON body: " + e.getMessage());
            runOnUiThread(() -> responseText.setText("Error: Unable to create request."));
            return;
        }

        Log.d(TAG, "Request JSON: " + jsonBody.toString()); // Log the request JSON

        RequestBody body = RequestBody.create(jsonBody.toString(), JSON);
        Request request = new Request.Builder()
                .url(LLM_API_URL)
                .header("Authorization", "Bearer " + LLM_API_KEY)
                .post(body)
                .build();

        client.newCall(request).enqueue(new Callback() {
            @SuppressLint("SetTextI18n")
            @Override
            public void onFailure(@NonNull Call call, @NonNull IOException e) {
                Log.e(TAG, "Network error: " + e.getMessage());
                runOnUiThread(() -> responseText.setText("Error: Unable to connect to LLM."));
            }

            @Override
            public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException {
                String jsonResponse = response.body() != null ? response.body().string() : "Empty response";
                Log.d(TAG, "Response JSON: " + jsonResponse); // Log the response JSON

                if (response.isSuccessful()) {
                    String botResponse = parseLLMResponse(jsonResponse);
                    runOnUiThread(() -> {
                        responseText.setText(botResponse);
                        speak(botResponse);
                    });
                } else {
                    Log.e(TAG, "Invalid response from LLM: " + response.code() + ", Body: " + jsonResponse);
                    runOnUiThread(() -> responseText.setText("Error: Invalid response from LLM."));
                }
            }
        });
    }


    private String parseLLMResponse(String jsonResponse) {
        try {
            // Parse the JSON response to extract the generated text
            JSONObject jsonObject = new JSONObject(jsonResponse);
            JSONArray choices = jsonObject.optJSONArray("choices");
            if (choices != null && choices.length() > 0) {
                return choices.getJSONObject(0).optString("text", "No text available").trim();
            } else {
                return "No response text available";
            }
        } catch (Exception e) {
            Log.e(TAG, "Error parsing response: " + e.getMessage());
            return "Error parsing response";
        }
    }


    private void speak(String text) {
        if (text != null && !text.isEmpty()) {
            textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null, null);
        } else {
            Log.e(TAG, "Text to speak is empty or null");
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (textToSpeech != null) {
            textToSpeech.stop();
            textToSpeech.shutdown();
        }
        if (speechRecognizer != null) {
            speechRecognizer.destroy();
        }
    }

    private void requestAudioPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.RECORD_AUDIO}, 1);
        } else {
            micButton.setEnabled(true);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == 1 && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            micButton.setEnabled(true);
        } else {
            micButton.setEnabled(false);
            Toast.makeText(this, "Microphone permission is required for this app to work.", Toast.LENGTH_SHORT).show();
        }
    }
}
