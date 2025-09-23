import 'dart:io';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

class FirebaseChatService {
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  final FirebaseAuth _auth = FirebaseAuth.instance;

  // Get current user ID
  String? get userId => _auth.currentUser?.uid ?? 'anonymous_user';

  // Sign in anonymously if no user
  Future<void> ensureUserSignedIn() async {
    try {
      if (_auth.currentUser == null) {
        await _auth.signInAnonymously();
        print('Firebase Auth: Signed in anonymously');
      }
    } catch (e) {
      print('Firebase Auth not configured, using offline mode: $e');
      // App will continue without authentication
    }
  }

  // Save chat message to Firebase
  Future<bool> saveChatMessage({
    required String message,
    required bool isUser,
    String? imagePath,
    double? latitude,
    double? longitude,
  }) async {
    try {
      await ensureUserSignedIn();

      await _firestore.collection('chat_messages').add({
        'userId': userId,
        'message': message,
        'isUser': isUser,
        'imagePath': imagePath,
        'latitude': latitude,
        'longitude': longitude,
        'timestamp': FieldValue.serverTimestamp(),
        'deviceInfo': {
          'platform': Platform.operatingSystem,
          'version': Platform.operatingSystemVersion,
        },
      });
      print('✅ Message saved to Firebase');
      return true;
    } catch (e) {
      print('❌ Failed to save to Firebase: $e');
      return false;
    }
  }

  // Get chat history for current user
  Stream<QuerySnapshot> getChatHistory() {
    try {
      if (userId == null || userId == 'anonymous_user') {
        // Return empty stream if no proper user ID
        return const Stream.empty();
      }

      return _firestore
          .collection('chat_messages')
          .where('userId', isEqualTo: userId)
          .orderBy('timestamp', descending: false)
          .snapshots();
    } catch (e) {
      print('Error getting chat history: $e');
      return const Stream.empty();
    }
  }

  // Save AI diagnosis result
  Future<void> saveDiagnosisResult({
    required String userMessage,
    required String aiResponse,
    String? imagePath,
    double? latitude,
    double? longitude,
  }) async {
    await ensureUserSignedIn();

    try {
      await _firestore.collection('diagnosis_history').add({
        'userId': userId,
        'userMessage': userMessage,
        'aiResponse': aiResponse,
        'imagePath': imagePath,
        'latitude': latitude,
        'longitude': longitude,
        'timestamp': FieldValue.serverTimestamp(),
        'deviceInfo': {
          'platform': Platform.operatingSystem,
          'version': Platform.operatingSystemVersion,
        },
      });
    } catch (e) {
      print('Error saving diagnosis to Firebase: $e');
    }
  }

  // Get diagnosis history
  Stream<QuerySnapshot> getDiagnosisHistory() {
    if (userId == null) return const Stream.empty();

    return _firestore
        .collection('diagnosis_history')
        .where('userId', isEqualTo: userId)
        .orderBy('timestamp', descending: true)
        .snapshots();
  }

  // Delete all chat data for current user
  Future<void> clearChatHistory() async {
    if (userId == null) return;

    try {
      // Delete chat messages
      var chatMessages =
          await _firestore
              .collection('chat_messages')
              .where('userId', isEqualTo: userId)
              .get();

      for (var doc in chatMessages.docs) {
        await doc.reference.delete();
      }

      // Delete diagnosis history
      var diagnosisHistory =
          await _firestore
              .collection('diagnosis_history')
              .where('userId', isEqualTo: userId)
              .get();

      for (var doc in diagnosisHistory.docs) {
        await doc.reference.delete();
      }
    } catch (e) {
      print('Error clearing chat history: $e');
    }
  }
}
