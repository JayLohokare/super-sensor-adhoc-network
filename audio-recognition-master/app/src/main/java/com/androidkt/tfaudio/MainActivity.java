package com.androidkt.tfaudio;

import android.Manifest;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.design.widget.Snackbar;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.SwitchCompat;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.gms.common.ConnectionResult;
import com.google.android.gms.common.api.GoogleApiClient;
import com.google.android.gms.common.api.ResultCallback;
import com.google.android.gms.common.api.Status;
import com.google.android.gms.nearby.Nearby;
import com.google.android.gms.nearby.messages.Message;
import com.google.android.gms.nearby.messages.MessageListener;
import com.google.android.gms.nearby.messages.PublishCallback;
import com.google.android.gms.nearby.messages.PublishOptions;
import com.google.android.gms.nearby.messages.Strategy;
import com.google.android.gms.nearby.messages.SubscribeCallback;
import com.google.android.gms.nearby.messages.SubscribeOptions;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.UUID;
import java.util.concurrent.locks.ReentrantLock;


public class MainActivity extends AppCompatActivity implements SensorEventListener ,GoogleApiClient.ConnectionCallbacks,
        GoogleApiClient.OnConnectionFailedListener {

    private static final int SAMPLE_RATE = 16000;
    private static final int SAMPLE_DURATION_MS = 1000;
    private static final int RECORDING_LENGTH = (int) (SAMPLE_RATE * SAMPLE_DURATION_MS / 1000);
    private static final long AVERAGE_WINDOW_DURATION_MS = 500;
    private static final float DETECTION_THRESHOLD = 0.2f; //Worked on 0.25
    private static final int SUPPRESSION_MS = 1500;
    private static final int MINIMUM_COUNT = 3;

    private static final long MINIMUM_TIME_BETWEEN_SAMPLES_MS = 30;
    private static final String LABEL_FILENAME = "file:///android_asset/conv_actions_labels.txt";
    private static final String MODEL_FILENAME = "file:///android_asset/new_model.pb";
    private static final String INPUT_DATA_NAME = "decoded_sample_data:0";
    private static final String SAMPLE_RATE_NAME = "decoded_sample_data:1";
    private static final String OUTPUT_SCORES_NAME = "labels_softmax";

    // UI elements.
    private static final int REQUEST_RECORD_AUDIO = 13;
    private static final String LOG_TAG = MainActivity.class.getSimpleName();
    private final ReentrantLock recordingBufferLock = new ReentrantLock();
    // Working variables.
    short[] recordingBuffer = new short[RECORDING_LENGTH];
    int recordingOffset = 0;
    boolean shouldContinue = true;
    boolean shouldContinueRecognition = true;
    private Button quitButton;
    private ListView labelsListView;
    private Thread recordingThread;
    private Thread recognitionThread;
    private TensorFlowInferenceInterface inferenceInterface;
    private List<String> labels = new ArrayList<String>();
    private List<String> displayedLabels = new ArrayList<>();
    private RecognizeCommands recognizeCommands = null;
    private List<Double> buffer;
    private Double variance;
    private TextView label;
    private TextView value;
    public static DecimalFormat DECIMAL_FORMATTER;
    private long prediction;

    private static final String TAG = MainActivity.class.getSimpleName();

    private static final int TTL_IN_SECONDS = 3 * 60; // Three minutes.

    // Key used in writing to and reading from SharedPreferences.
    private static final String KEY_UUID = "key_uuid";

    /**
     * Sets the time in seconds for a published message or a subscription to live. Set to three
     * minutes in this sample.
     */
    private static final Strategy PUB_SUB_STRATEGY = new Strategy.Builder()
            .setTtlSeconds(TTL_IN_SECONDS).build();

    private static final String[] DANGEROUS_PERMISSIONS = {
            Manifest.permission.READ_PHONE_STATE,
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.INTERNET,
            Manifest.permission.ACCESS_COARSE_LOCATION,
            Manifest.permission.BLUETOOTH,
            Manifest.permission.BLUETOOTH_ADMIN,
            Manifest.permission.ACCESS_WIFI_STATE,
            Manifest.permission.CHANGE_WIFI_STATE
    };


    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        for (int grantResult : grantResults) {
            // handle denied permissions
        }

        buildGoogleApiClient();
        startRecording();
        SensorManager sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        Sensor magnetometer = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);
        sensorManager.registerListener(this, magnetometer, SensorManager.SENSOR_DELAY_NORMAL);
        startRecognition();
    }


    private void initPermissions() {
        List<String> missingPermissions = new ArrayList<String>();
        for (String permission : DANGEROUS_PERMISSIONS) {
            if (ActivityCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                missingPermissions.add(permission);
            }
        }

        if (missingPermissions.size() > 0) {
            String[] permissions = new String[missingPermissions.size()];
            ActivityCompat.requestPermissions(
                    this,
                    missingPermissions.toArray(permissions),
                    1);
        } else {
            // we have all permissions, move on
        }
    }


    private static String getUUID(SharedPreferences sharedPreferences) {
        String uuid = sharedPreferences.getString(KEY_UUID, "");
        if (TextUtils.isEmpty(uuid)) {
            uuid = UUID.randomUUID().toString();
            sharedPreferences.edit().putString(KEY_UUID, uuid).apply();
        }
        return uuid;
    }

    /**
     * The entry point to Google Play Services.
     */
    private GoogleApiClient mGoogleApiClient;

    // Views.
    private SwitchCompat mPublishSwitch;
    private SwitchCompat mSubscribeSwitch;



    private Button btn;
    //private EditText edttxt;
    /**
     * The {@link Message} object used to broadcast information about the device to nearby devices.
     */
    private Message mPubMessage;

    /**
     * A {@link MessageListener} for processing messages from nearby devices.
     */
    private MessageListener mMessageListener;

    /**
     * Adapter for working with messages from nearby publishers.
     */
    private ArrayAdapter<String> mNearbyDevicesArrayAdapter;


    public static boolean hasPermissions(Context context, String... permissions) {
        if (android.os.Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && context != null && permissions != null) {
            for (String permission : permissions) {
                if (ActivityCompat.checkSelfPermission(context, permission) != PackageManager.PERMISSION_GRANTED) {
                    return false;
                }
            }
        }
        return true;
    }


    @Override
    protected void onCreate(Bundle savedInstanceState)
    {

        DecimalFormatSymbols symbols = new DecimalFormatSymbols(Locale.US);
        symbols.setDecimalSeparator('.');
        DECIMAL_FORMATTER = new DecimalFormat("#.000", symbols);
        // Set up the UI.
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main_layout);

        String[] PERMISSIONS = {

                Manifest.permission.RECORD_AUDIO,
                Manifest.permission.READ_PHONE_STATE,
                Manifest.permission.READ_SMS,
                Manifest.permission.READ_EXTERNAL_STORAGE,
                Manifest.permission.ACCESS_COARSE_LOCATION,
                Manifest.permission.BLUETOOTH,
                Manifest.permission.BLUETOOTH_ADMIN,
                Manifest.permission.ACCESS_WIFI_STATE,
                Manifest.permission.CHANGE_WIFI_STATE
        };

        if(!hasPermissions(this, PERMISSIONS)){
            ActivityCompat.requestPermissions(this, PERMISSIONS, 1);
        }







        String actualFilename = LABEL_FILENAME.split("file:///android_asset/")[1];
        Log.i(LOG_TAG, "Reading labels from: " + actualFilename);
        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(getAssets().open(actualFilename)));
            String line;
            while ((line = br.readLine()) != null) {
                labels.add(line);
                if (line.charAt(0) != '_') {
                    displayedLabels.add(line.substring(0, 1).toUpperCase() + line.substring(1));
                }
            }
            br.close();
        } catch (IOException e) {
            throw new RuntimeException("Problem reading label file!", e);
        }


        // Set up an object to smooth recognition results to increase accuracy.
        recognizeCommands =
                new RecognizeCommands(
                        labels,
                        AVERAGE_WINDOW_DURATION_MS,
                        DETECTION_THRESHOLD,
                        SUPPRESSION_MS,
                        MINIMUM_COUNT,
                        MINIMUM_TIME_BETWEEN_SAMPLES_MS);

        // Load the TensorFlow model.
        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILENAME);



        mSubscribeSwitch = (SwitchCompat) findViewById(R.id.subscribe_switch);
        mPublishSwitch = (SwitchCompat) findViewById(R.id.publish_switch);

        //edttxt = (EditText) findViewById(R.id.edttxt);
        btn = (Button) findViewById(R.id.button);


        btn.setOnClickListener(new View.OnClickListener() {
                                   @Override
                                   public void onClick(View view) {
                                       subscribe();
                                       String text = Long.toString(prediction);
                                       Log.d("publishing" , Long.toString(prediction));
                                       mPubMessage = DeviceMessage.newNearbyMessage(getUUID(getSharedPreferences(
                                               getApplicationContext().getPackageName(), Context.MODE_PRIVATE)), text, 5);
                                       publish();



                                   }
                               }
        );

        // Build the message that is going to be published. This contains the device name and a
        // UUID.

        mMessageListener = new MessageListener() {
            @Override
            public void onFound(final Message message) {
                // Called when a new message is found.

                Toast.makeText(MainActivity.this, DeviceMessage.fromNearbyMessage(message).getmUUID(), Toast.LENGTH_SHORT).show();
                Toast.makeText(MainActivity.this, getUUID(getSharedPreferences(
                        getApplicationContext().getPackageName(), Context.MODE_PRIVATE)), Toast.LENGTH_SHORT).show();

                if (DeviceMessage.fromNearbyMessage(message).getmUUID().equals(getUUID(getSharedPreferences(
                        getApplicationContext().getPackageName(), Context.MODE_PRIVATE))) || DeviceMessage.fromNearbyMessage(message).getmUUID() == (getUUID(getSharedPreferences(
                        getApplicationContext().getPackageName(), Context.MODE_PRIVATE))) || DeviceMessage.fromNearbyMessage(message).getHopCount() == 0) {

                } else {

                    mNearbyDevicesArrayAdapter.add(DeviceMessage.fromNearbyMessage(message).getMessageBody());
                    mPubMessage = DeviceMessage.newNearbyMessage(DeviceMessage.fromNearbyMessage(message).getmUUID(), DeviceMessage.fromNearbyMessage(message).getMessageBody(), DeviceMessage.fromNearbyMessage(message).getHopCount() - 1);
                    publish();
                }


            }



            @Override
            public void onLost(final Message message) {
                // Called when a message is no longer detectable nearby.
                mNearbyDevicesArrayAdapter.remove(
                        DeviceMessage.fromNearbyMessage(message).getMessageBody());
            }
        };

        mSubscribeSwitch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                // If GoogleApiClient is connected, perform sub actions in response to user action.
                // If it isn't connected, do nothing, and perform sub actions when it connects (see
                // onConnected()).
                if (mGoogleApiClient != null && mGoogleApiClient.isConnected()) {
                    if (isChecked) {
                        subscribe();
                    } else {
                        unsubscribe();
                    }
                }
            }
        });

        mPublishSwitch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                // If GoogleApiClient is connected, perform pub actions in response to user action.
                // If it isn't connected, do nothing, and perform pub actions when it connects (see
                // onConnected()).
                if (mGoogleApiClient != null && mGoogleApiClient.isConnected()) {
                    if (isChecked) {
                        //publish();
                    } else {
                        //unpublish();
                    }
                }
            }
        });

        final List<String> nearbyDevicesArrayList = new ArrayList<>();
        mNearbyDevicesArrayAdapter = new ArrayAdapter<>(this,
                android.R.layout.simple_list_item_1,
                nearbyDevicesArrayList);
        final ListView nearbyDevicesListView = (ListView) findViewById(
                R.id.nearby_devices_list_view);
        if (nearbyDevicesListView != null)
        {
            nearbyDevicesListView.setAdapter(mNearbyDevicesArrayAdapter);
        }

        //here
//        subscribe();
//        String text = Long.toString(prediction);
//        mPubMessage = DeviceMessage.newNearbyMessage(getUUID(getSharedPreferences(
//                getApplicationContext().getPackageName(), Context.MODE_PRIVATE)), text, 5);
//        publish();


    }




    public synchronized void startRecording() {


        buffer = new ArrayList<Double>();
        if (recordingThread != null) {
            return;
        }
        shouldContinue = true;
        recordingThread =
                new Thread(
                        new Runnable() {
                            @Override
                            public void run() {
                                record();
                            }
                        });
        recordingThread.start();
    }

    public synchronized void stopRecording() {

        if (recordingThread == null) {
            return;
        }
        shouldContinue = false;
        recordingThread = null;
    }


    private void record() {
        android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO);

        // Estimate the buffer size we'll need for this device.
        int bufferSize =
                AudioRecord.getMinBufferSize(
                        SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
            bufferSize = SAMPLE_RATE * 2;
        }
        short[] audioBuffer = new short[bufferSize / 2];

        AudioRecord record =
                new AudioRecord(
                        MediaRecorder.AudioSource.DEFAULT,
                        SAMPLE_RATE,
                        AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT,
                        bufferSize);

        if (record.getState() != AudioRecord.STATE_INITIALIZED) {
            Log.e(LOG_TAG, "Audio Record can't initialize!");
            return;
        }

        record.startRecording();

        Log.v(LOG_TAG, "Start recording");

        // Loop, gathering audio data and copying it to a round-robin buffer.
        while (shouldContinue) {
            int numberRead = record.read(audioBuffer, 0, audioBuffer.length);
            int maxLength = recordingBuffer.length;
            int newRecordingOffset = recordingOffset + numberRead;
            int secondCopyLength = Math.max(0, newRecordingOffset - maxLength);
            int firstCopyLength = numberRead - secondCopyLength;
            // We store off all the data for the recognition thread to access. The ML
            // thread will copy out of this buffer into its own, while holding the
            // lock, so this should be thread safe.
            recordingBufferLock.lock();
            try {
                System.arraycopy(audioBuffer, 0, recordingBuffer, recordingOffset, firstCopyLength);
                System.arraycopy(audioBuffer, firstCopyLength, recordingBuffer, 0, secondCopyLength);
                recordingOffset = newRecordingOffset % maxLength;
            } finally {
                recordingBufferLock.unlock();
            }
        }

        record.stop();
        record.release();
    }

    public synchronized void startRecognition() {
        if (recognitionThread != null) {
            return;
        }
        shouldContinueRecognition = true;
        recognitionThread =
                new Thread(
                        new Runnable() {
                            @Override
                            public void run() {
                                recognize();
                            }
                        });
        recognitionThread.start();
    }

    public synchronized void stopRecognition() {
        if (recognitionThread == null) {
            return;
        }
        shouldContinueRecognition = false;
        recognitionThread = null;
    }


    private void recognize() {
        Log.v(LOG_TAG, "Start recognition");

        short[] inputBuffer = new short[RECORDING_LENGTH];
        float[] floatInputBuffer = new float[RECORDING_LENGTH];
        float[] outputScores = new float[labels.size()];
        String[] outputScoresNames = new String[]{OUTPUT_SCORES_NAME};
        int[] sampleRateList = new int[]{SAMPLE_RATE};

        // Loop, grabbing recorded data and running the recognition model on it.
        while (shouldContinueRecognition) {
            // The recording thread places data in this round-robin buffer, so lock to
            // make sure there's no writing happening and then copy it to our own
            // local version.
            recordingBufferLock.lock();
            try {
                int maxLength = recordingBuffer.length;
                int firstCopyLength = maxLength - recordingOffset;
                int secondCopyLength = recordingOffset;
                System.arraycopy(recordingBuffer, recordingOffset, inputBuffer, 0, firstCopyLength);
                System.arraycopy(recordingBuffer, 0, inputBuffer, firstCopyLength, secondCopyLength);
            } finally {
                recordingBufferLock.unlock();
            }

            // We need to feed in float values between -1.0f and 1.0f, so divide the
            // signed 16-bit inputs.
            for (int i = 0; i < RECORDING_LENGTH; ++i) {
                floatInputBuffer[i] = inputBuffer[i] / 32767.0f;
            }

            // Run the model.
            inferenceInterface.feed(SAMPLE_RATE_NAME, sampleRateList);
            inferenceInterface.feed(INPUT_DATA_NAME, floatInputBuffer, RECORDING_LENGTH, 1);
            inferenceInterface.run(outputScoresNames);
            inferenceInterface.fetch(OUTPUT_SCORES_NAME, outputScores);

            // Use the smoother to figure out if we've had a real recognition event.
            long currentTime = System.currentTimeMillis();
            final RecognizeCommands.RecognitionResult result = recognizeCommands.processLatestResults(outputScores, currentTime);

            runOnUiThread(
                    new Runnable() {
                        @Override
                        public void run() {
                            // If we do have a new command, highlight the right list entry.
                            if (!result.foundCommand.startsWith("_") && result.isNewCommand) {
                                int labelIndex = -1;
                                for (int i = 0; i < labels.size(); ++i) {
                                    if (labels.get(i).equals(result.foundCommand))
                                    {
                                        labelIndex = i;
                                    }
                                }
                                if (result.foundCommand.equalsIgnoreCase("newmicrowave") && variance > 100)

                                {

                                    prediction = 1;
                                }
                                else if (result.foundCommand.equalsIgnoreCase("newmicrowave") && (variance <= 100 || variance.isNaN())) {
                                    prediction = 1000;
                                }
                                else
                                    {
                                    if (result.foundCommand.equalsIgnoreCase("tap")) {
                                        prediction = 2;
                                    } else

                                    {
                                        prediction = 1000000;
                                    }

                                }
                            } else if (result.isNewCommand && result.foundCommand.startsWith("_"))
                            {
                                prediction = 0;
                                //label.setText(result.foundCommand);
                            } else {

                            }
                            //Toast.makeText(MainActivity.this, Long.toString(prediction), Toast.LENGTH_SHORT).show();
                            Log.d("Predicted", Long.toString(prediction));
                        }
                    });
            try {
                // We don't need to run too frequently, so snooze for a bit.
                Thread.sleep(MINIMUM_TIME_BETWEEN_SAMPLES_MS);
            } catch (InterruptedException e) {
                // Ignore
            }
        }

        Log.v(LOG_TAG, "End recognition");
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        System.out.println("Sensed magnetic");
        if (event.sensor.getType() == Sensor.TYPE_MAGNETIC_FIELD) {
            // get values for each axes X,Y,Z
            float magX = event.values[0];
            float magY = event.values[1];
            float magZ = event.values[2];
            double magnitude = Math.sqrt((magX * magX) + (magY * magY) + (magZ * magZ));
            // set value on the screen
            buffer.add(magnitude);
            double sum = 0.0;
            for (double a : buffer)
                sum += a;
            double mean = sum / (buffer.size());
            double temp = 0;
            for (double a : buffer)
                temp += (a - mean) * (a - mean);
            variance = temp / (buffer.size() - 1);
        }

    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }



    private void buildGoogleApiClient() {
        if (mGoogleApiClient != null) {
            return;
        }
        mGoogleApiClient = new GoogleApiClient.Builder(this)
                .addApi(Nearby.MESSAGES_API)
                .addConnectionCallbacks(this)
                .enableAutoManage(this, this)
                .build();
    }

    @Override
    public void onConnectionFailed(@NonNull ConnectionResult connectionResult) {
        mPublishSwitch.setEnabled(false);
        mSubscribeSwitch.setEnabled(false);
        logAndShowSnackbar("Exception while connecting to Google Play services: " +
                connectionResult.getErrorMessage());
    }

    @Override
    public void onConnectionSuspended(int i) {
        logAndShowSnackbar("Connection suspended. Error code: " + i);
    }

    @Override
    public void onConnected(@Nullable Bundle bundle) {
        Log.i(TAG, "GoogleApiClient connected");
        // We use the Switch buttons in the UI to track whether we were previously doing pub/sub (
        // switch buttons retain state on orientation change). Since the GoogleApiClient disconnects
        // when the activity is destroyed, foreground pubs/subs do not survive device rotation. Once
        // this activity is re-created and GoogleApiClient connects, we check the UI and pub/sub
        // again if necessary.
        if (mPublishSwitch.isChecked()) {
            publish();
        }
        if (mSubscribeSwitch.isChecked()) {
            subscribe();
        }
    }

    /**
     * Subscribes to messages from nearby devices and updates the UI if the subscription either
     * fails or TTLs.
     */
    private void subscribe() {
        Log.i(TAG, "Subscribing");
        mNearbyDevicesArrayAdapter.clear();
        SubscribeOptions options = new SubscribeOptions.Builder()
                .setStrategy(PUB_SUB_STRATEGY)
                .setCallback(new SubscribeCallback() {
                    @Override
                    public void onExpired() {
                        super.onExpired();
                        Log.i(TAG, "No longer subscribing");
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                mSubscribeSwitch.setChecked(false);
                            }
                        });
                    }
                }).build();

        Nearby.Messages.subscribe(mGoogleApiClient, mMessageListener, options)
                .setResultCallback(new ResultCallback<Status>() {
                    @Override
                    public void onResult(@NonNull Status status) {
                        if (status.isSuccess()) {
                            Log.i(TAG, "Subscribed successfully.");
                        } else {
                            logAndShowSnackbar("Could not subscribe, status = " + status);
                            mSubscribeSwitch.setChecked(false);
                        }
                    }
                });
    }

    /**
     * Publishes a message to nearby devices and updates the UI if the publication either fails or
     * TTLs.
     */
    private void publish() {
        Log.i(TAG, "Publishing");
        PublishOptions options = new PublishOptions.Builder()
                .setStrategy(PUB_SUB_STRATEGY)
                .setCallback(new PublishCallback() {
                    @Override
                    public void onExpired() {
                        super.onExpired();
                        Log.i(TAG, "No longer publishing");
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                mPublishSwitch.setChecked(false);
                            }
                        });
                    }
                }).build();

        Nearby.Messages.publish(mGoogleApiClient, mPubMessage, options)
                .setResultCallback(new ResultCallback<Status>() {
                    @Override
                    public void onResult(@NonNull Status status) {
                        if (status.isSuccess()) {
                            Log.i(TAG, "Published successfully.");
                        } else {
                            logAndShowSnackbar("Could not publish, status = " + status);
                            mPublishSwitch.setChecked(false);
                        }
                    }
                });
    }

    /**
     * Stops subscribing to messages from nearby devices.
     */
    private void unsubscribe() {
        Log.i(TAG, "Unsubscribing.");
        Nearby.Messages.unsubscribe(mGoogleApiClient, mMessageListener);
    }

    /**
     * Stops publishing message to nearby devices.
     */
    private void unpublish() {
        Log.i(TAG, "Unpublishing.");
        Nearby.Messages.unpublish(mGoogleApiClient, mPubMessage);
    }

    /**
     * Logs a message and shows a {@link Snackbar} using {@code text};
     *
     * @param text The text used in the Log message and the SnackBar.
     */
    private void logAndShowSnackbar(final String text) {
        Log.w(TAG, text);
        View container = findViewById(R.id.activity_main_container);
        if (container != null) {
            Snackbar.make(container, text, Snackbar.LENGTH_LONG).show();
        }
    }
}