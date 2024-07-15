package com.example.espark_1;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.media.RingtoneManager;
import android.net.Uri;
import android.os.Build;
import android.util.Log;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;
import androidx.core.app.NotificationCompat;
import androidx.core.app.NotificationManagerCompat;
import androidx.core.content.ContextCompat;

import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.database.annotations.NotNull;
import com.google.firebase.messaging.FirebaseMessagingService;
import com.google.firebase.messaging.RemoteMessage;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class MyFirebaseMessagingService extends FirebaseMessagingService {

    private static final String CHANNEL_ID = "Notification_channel";
    private static int count = 0;

    /*
     * This is automatically called when notification is being received
     * */
    @Override
    public void onMessageReceived(@NonNull @NotNull RemoteMessage remoteMessage) {
        super.onMessageReceived(remoteMessage);

        // Check if message contains a notification payload
        if (remoteMessage.getNotification() != null) {
            // Create and show notification
            showNotification(remoteMessage.getNotification().getTitle(), remoteMessage.getNotification().getBody(), remoteMessage);
        }
    }


    private void showNotification(String title, String message,RemoteMessage remoteMessage) {
        // Notification channel ID is required for Android Oreo and higher
        String channelId = "default_channel_id";
        String channelName = "Default Channel";

        NotificationManager notificationManager = getSystemService(NotificationManager.class);

        // Create notification channel if the SDK version is >= Android Oreo
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(channelId, channelName, NotificationManager.IMPORTANCE_DEFAULT);
            notificationManager.createNotificationChannel(channel);
        }

      /*  Map<String,String> opendata = remoteMessage.getData();
        String actionValue = opendata.get("openactivity");
        Intent intent=new Intent();
        switch(actionValue){
            case "BROWSER":
                intent=new Intent(this, LoginActivity.class);
                break;
            case "HOME":
                //redirect to another activity
                break;
            case "PAYMENT":
                break;
        }

        intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
        //intent.putExtra("pushnotification","True");
        PendingIntent pendingIntent = PendingIntent.getActivity(this, 0, intent, PendingIntent.FLAG_ONE_SHOT);

       /* Uri defaultSoundUri = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_NOTIFICATION);
        NotificationManager mNotifyManager = (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            int importance = NotificationManager.IMPORTANCE_LOW;
            NotificationChannel mChannel = new NotificationChannel("MyID", "Myapp", importance);
            mChannel.setDescription(remoteMessage.getData().get("message"));
            mChannel.enableLights(true);
            mChannel.setLightColor(R.color.red);
            mChannel.enableVibration(true);
            mNotifyManager.createNotificationChannel(mChannel);
        }
*/
        // Create intent to open a specific activity when notification is clicked
 /*       Intent intent = new Intent(this, Tutorial4Activity.class);
        intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP); // If activity is already running, bring it to the front
        PendingIntent pendingIntent = PendingIntent.getActivity(this, 0, intent, PendingIntent.FLAG_ONE_SHOT);
*/
        Intent intent = new Intent("OPEN_ACTIVITY_2");
       // Intent intent = new Intent(/*getApplicationContext(), Tutorial4Activity.class*/);
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK); // If activity is already running, bring it to the front
        PendingIntent pendingIntent = PendingIntent.getActivity(getApplicationContext(), 1, intent, PendingIntent.FLAG_ONE_SHOT);
        //PendingIntent pendingIntent = PendingIntent.getBroadcast(this, 0, intent, PendingIntent.FLAG_UPDATE_CURRENT);

        // Build notification
        NotificationCompat.Builder builder = new NotificationCompat.Builder(this, channelId)
                .setContentTitle(title)
                .setContentText(message)
                .setSmallIcon(R.drawable.ic_notification_icon)
                .setAutoCancel(true) // Dismiss the notification when clicked
                .setContentIntent(pendingIntent);
                /*.setSound(defaultSoundUri)
                .setColor(getResources().getColor(R.color.blue))
                .setContentIntent(pendingIntent)
                .setChannelId("Myid")
                .setPriority(NotificationCompat.PRIORITY_HIGH);*/

        // Show notification
        notificationManager.notify(0, builder.build());
        count ++;
    }

}
