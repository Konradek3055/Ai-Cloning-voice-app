package pl.kinotts.studio.background

import android.Manifest
import android.app.Activity
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import androidx.core.app.ActivityCompat
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob

/**
 * Utrzymuje proces przy życiu podczas długich zadań (synteza, pobieranie modeli).
 * Zadania uruchamiane w [appScope] nie są anulowane po wyjściu z aplikacji ani po zamknięciu ekranu,
 * a usługa pierwszoplanowa z powiadomieniem chroni proces przed zabiciem przez system.
 */
object KeepAlive {
    val appScope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)

    private val lock = Any()
    private var active = 0
    private var startedAt = 0L
    @Volatile private var title = ""
    private val idleActions = ArrayList<() -> Unit>()
    private val handler = Handler(Looper.getMainLooper())

    fun currentTitle(): String = title

    fun begin(context: Context, text: String) {
        val app = context.applicationContext
        var startService = false
        synchronized(lock) {
            active++
            title = text
            if (active == 1) {
                startService = true
                startedAt = System.currentTimeMillis()
            }
        }
        if (Build.VERSION.SDK_INT >= 33 && context is Activity &&
            ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            runCatching {
                ActivityCompat.requestPermissions(context, arrayOf(Manifest.permission.POST_NOTIFICATIONS), 4711)
            }
        }
        if (startService) {
            runCatching { ContextCompat.startForegroundService(app, Intent(app, KeepAliveService::class.java)) }
        }
    }

    fun end(context: Context) {
        val app = context.applicationContext
        var idle = false
        var actions: List<() -> Unit> = emptyList()
        synchronized(lock) {
            if (active > 0) active--
            if (active == 0) {
                idle = true
                actions = ArrayList(idleActions)
                idleActions.clear()
            }
        }
        if (idle) {
            // Android wymaga, by usługa zdążyła wywołać startForeground() - nie zatrzymujemy jej od razu.
            val wait = (1500L - (System.currentTimeMillis() - startedAt)).coerceAtLeast(0L)
            handler.postDelayed({
                val stillIdle = synchronized(lock) { active == 0 }
                if (stillIdle) runCatching { app.stopService(Intent(app, KeepAliveService::class.java)) }
            }, wait)
            actions.forEach { runCatching { it() } }
        }
    }

    /** Wykonuje akcję od razu, gdy nic nie pracuje w tle, albo dopiero po zakończeniu bieżących zadań. */
    fun runWhenIdle(action: () -> Unit) {
        val runNow = synchronized(lock) {
            if (active == 0) true else { idleActions.add(action); false }
        }
        if (runNow) action()
    }
}

class KeepAliveService : Service() {
    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val manager = getSystemService(NotificationManager::class.java)
        if (manager.getNotificationChannel(CHANNEL_ID) == null) {
            manager.createNotificationChannel(
                NotificationChannel(CHANNEL_ID, "KinoTTS w tle", NotificationManager.IMPORTANCE_LOW)
            )
        }
        val openApp = packageManager.getLaunchIntentForPackage(packageName)?.let { launch ->
            PendingIntent.getActivity(this, 0, launch, PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        }
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setContentTitle("KinoTTS Studio")
            .setContentText(KeepAlive.currentTitle().ifBlank { "Trwa praca w tle…" })
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setContentIntent(openApp)
            .build()
        ServiceCompat.startForeground(this, NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        stopForeground(STOP_FOREGROUND_REMOVE)
        super.onDestroy()
    }

    private companion object {
        const val CHANNEL_ID = "kinotts_background"
        const val NOTIFICATION_ID = 4711
    }
}
