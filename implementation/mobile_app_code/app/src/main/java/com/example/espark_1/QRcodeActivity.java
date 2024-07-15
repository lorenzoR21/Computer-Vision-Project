package com.example.espark_1;

import androidx.activity.result.ActivityResultLauncher;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import com.google.zxing.integration.android.IntentIntegrator;
import com.google.zxing.integration.android.IntentResult;
import com.journeyapps.barcodescanner.CaptureActivity;
import com.journeyapps.barcodescanner.ScanContract;
import com.journeyapps.barcodescanner.ScanOptions;

public class QRcodeActivity extends AppCompatActivity  {

    private static final int PERMISSION_REQUEST_CAMERA = 1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        //setContentView(R.layout.activity_qrcode);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, new String[]{android.Manifest.permission.CAMERA}, PERMISSION_REQUEST_CAMERA);
            } else {
                initQRCodeScanner();
            }
        } else {
            initQRCodeScanner();
        }
    }

    // Register the launcher and result handler
    private final ActivityResultLauncher<ScanOptions> barcodeLauncher = registerForActivityResult(new ScanContract(),
            result -> {
                if(result.getContents() == null) {
                    Toast.makeText(QRcodeActivity.this, "Cancelled", Toast.LENGTH_LONG).show();
                } else {
                    Toast.makeText(QRcodeActivity.this, "Scanned", Toast.LENGTH_LONG).show();
                    String c = result.getContents();
                    String[] substrings = c.split(";");
                    if (substrings[0].equals("info")) //QR code of parking information
                    {
                        Intent intent = new Intent(QRcodeActivity.this, ParkingActivity.class);
                        intent.putExtra("parking", substrings[1]);
                        startActivity(intent);
                    }
                    else if (substrings[0].equals("pay")) //QR code of payment page
                    {
                        Intent intent = new Intent(QRcodeActivity.this, PayActivity.class);
                        intent.putExtra("id", substrings[1]);
                        intent.putExtra("name", substrings[2]);
                        intent.putExtra("costH", Float.parseFloat(substrings[3]));
                        startActivity(intent);
                    }
                    else
                    {
                        Toast.makeText(QRcodeActivity.this, "Cancelled", Toast.LENGTH_LONG).show();
                    }
                }
            });

    private void initQRCodeScanner() {
        ScanOptions options = new ScanOptions();
        options.setDesiredBarcodeFormats(ScanOptions.QR_CODE);
        options.setCaptureActivity(CustomCaptureActivity.class); //use a custom class CustomCaptureActivity to change the options of the scanning screen
        options.setPrompt("Scan the QR code to go to the payment page or to see parking information");
        options.setOrientationLocked(false);
        options.setBeepEnabled(false);
        options.setBarcodeImageEnabled(true);
        barcodeLauncher.launch(options);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSION_REQUEST_CAMERA) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                initQRCodeScanner();
            } else {
                Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }
}