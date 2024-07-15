package com.example.espark_1;

import android.content.Context;
import android.content.Intent;
import android.os.AsyncTask;
import android.util.Log;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

public class NewsTask extends AsyncTask<Void, Void, String> {

    private Context mContext;

    public NewsTask(Context context) {
        mContext = context;
    }
    @Override
    protected String doInBackground(Void... voids) {
        String result = null;
        HttpURLConnection urlConnection = null;
        try {
            URL url = new URL("http://sor1.pythonanywhere.com");
            urlConnection = (HttpURLConnection) url.openConnection();
            BufferedReader reader = new BufferedReader(new InputStreamReader(urlConnection.getInputStream()));
            StringBuilder stringBuilder = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                stringBuilder.append(line).append("\n");
            }
            result = stringBuilder.toString();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (urlConnection != null) {
                urlConnection.disconnect();
            }
        }
        return result;
    }

    @Override
    protected void onPostExecute(String result) {
        if (result != null) {
            // Handle the result, for example, parse the JSON response
            Intent intent = new Intent(mContext, NewsActivity.class);
            intent.putExtra("news", result);
            mContext.startActivity(intent);
            Log.d("FetchNewsTask", result);
        } else {
            Log.e("FetchNewsTask", "Failed to fetch news");
        }
    }
}
