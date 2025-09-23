import 'dart:io';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'api_service.dart';
import 'firebase_chat_service.dart';

class ChatMessage {
  final String text;
  final File? image;
  final bool isUser;
  final DateTime timestamp;

  ChatMessage({
    required this.text,
    this.image,
    required this.isUser,
    required this.timestamp,
  });
}

class ChatScreen extends StatefulWidget {
  final String? initialMessage;
  final File? initialImage;

  const ChatScreen({super.key, this.initialMessage, this.initialImage});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> with TickerProviderStateMixin {
  File? _selectedImage;
  final ImagePicker _picker = ImagePicker();
  final TextEditingController _textController = TextEditingController();
  final List<ChatMessage> _messages = [];
  final ScrollController _scrollController = ScrollController();
  final FirebaseChatService _firebaseService = FirebaseChatService();
  late AnimationController _animationController;
  late Animation<double> _fadeAnimation;
  late Animation<Offset> _slideAnimation;

  @override
  void initState() {
    super.initState();

    // Setup animations
    _animationController = AnimationController(
      duration: const Duration(milliseconds: 800),
      vsync: this,
    );

    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _animationController, curve: Curves.easeInOut),
    );

    _slideAnimation = Tween<Offset>(
      begin: const Offset(0, 0.5),
      end: Offset.zero,
    ).animate(
      CurvedAnimation(parent: _animationController, curve: Curves.easeOutCubic),
    );

    // Start animation
    _animationController.forward();

    // Add initial message if provided
    if (widget.initialMessage != null || widget.initialImage != null) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        _addInitialMessage();
      });
    }
  }

  @override
  void dispose() {
    _animationController.dispose();
    _textController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  void _addInitialMessage() {
    // Set the initial text in the text controller
    if (widget.initialMessage != null) {
      _textController.text = widget.initialMessage!;
    }

    // Set the initial image
    if (widget.initialImage != null) {
      setState(() {
        _selectedImage = widget.initialImage;
      });
    }

    // Automatically send the message if there's an initial message or image
    if (widget.initialMessage != null || widget.initialImage != null) {
      // Add a small delay to ensure UI is ready
      Future.delayed(Duration(milliseconds: 500), () {
        _sendMessage();
      });
    }

    _scrollToBottom();
  }

  Future<void> _showImagePickerDialog() async {
    showModalBottomSheet(
      context: context,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (BuildContext context) {
        return SafeArea(
          child: Wrap(
            children: [
              ListTile(
                leading: Icon(Icons.photo_library),
                title: Text('Choose from Gallery'),
                onTap: () {
                  Navigator.pop(context);
                  _pickImage(ImageSource.gallery);
                },
              ),
              ListTile(
                leading: Icon(Icons.photo_camera),
                title: Text('Take a Photo'),
                onTap: () {
                  Navigator.pop(context);
                  _pickImage(ImageSource.camera);
                },
              ),
            ],
          ),
        );
      },
    );
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? pickedFile = await _picker.pickImage(
        source: source,
        maxWidth: 1800,
        maxHeight: 1800,
        imageQuality: 80,
      );

      if (pickedFile != null) {
        setState(() {
          _selectedImage = File(pickedFile.path);
        });

        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Image selected successfully!'),
            backgroundColor: Colors.green,
            duration: Duration(seconds: 2),
          ),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error selecting image: $e'),
          backgroundColor: Colors.red,
          duration: Duration(seconds: 3),
        ),
      );
    }
  }

  // Get local chat history for current session
  // Send chat history to endpoint
  Future<void> _sendHistoryToEndpoint(List<QueryDocumentSnapshot> docs) async {
    try {
      // Prepare history data
      List<Map<String, dynamic>> historyData =
          docs.map((doc) {
            var data = doc.data() as Map<String, dynamic>;
            return {
              'message': data['message'],
              'isUser': data['isUser'],
              'timestamp': data['timestamp']?.toDate()?.toIso8601String(),
              'latitude': data['latitude'],
              'longitude': data['longitude'],
            };
          }).toList();

      // Show loading dialog
      showDialog(
        context: context,
        barrierDismissible: false,
        builder:
            (context) => AlertDialog(
              content: Row(
                children: [
                  CircularProgressIndicator(),
                  SizedBox(width: 16),
                  Text('Sending history to AI model...'),
                ],
              ),
            ),
      );

      // Send to your endpoint with history
      String response = await ApiService.sendChatHistory(historyData);

      // Close loading dialog
      Navigator.pop(context);

      // Show result
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('History sent successfully!'),
          backgroundColor: Colors.green,
        ),
      );

      // Show AI response in a dialog
      showDialog(
        context: context,
        builder:
            (context) => AlertDialog(
              title: Text('AI Analysis of Chat History'),
              content: SingleChildScrollView(child: Text(response)),
              actions: [
                TextButton(
                  onPressed: () => Navigator.pop(context),
                  child: Text('Close'),
                ),
              ],
            ),
      );
    } catch (e) {
      // Close loading dialog if still open
      Navigator.pop(context);

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error sending history: $e'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  void _showChatHistory() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder:
          (context) => Container(
            height: MediaQuery.of(context).size.height * 0.8,
            padding: EdgeInsets.all(16),
            child: Column(
              children: [
                Container(
                  width: 40,
                  height: 4,
                  decoration: BoxDecoration(
                    color: Colors.grey[300],
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
                SizedBox(height: 16),
                Text(
                  'Chat History',
                  style: GoogleFonts.inter(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                SizedBox(height: 16),
                Expanded(
                  child: StreamBuilder(
                    stream: _firebaseService.getChatHistory(),
                    builder: (context, snapshot) {
                      if (snapshot.connectionState == ConnectionState.waiting) {
                        return Center(
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              CircularProgressIndicator(),
                              SizedBox(height: 16),
                              Text(
                                'Loading chat history...',
                                style: GoogleFonts.inter(color: Colors.grey),
                              ),
                            ],
                          ),
                        );
                      }

                      if (snapshot.hasError) {
                        return Center(
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(
                                Icons.error_outline,
                                size: 48,
                                color: Colors.red,
                              ),
                              SizedBox(height: 16),
                              Text(
                                'Error loading history',
                                style: GoogleFonts.inter(color: Colors.red),
                              ),
                            ],
                          ),
                        );
                      }

                      if (!snapshot.hasData ||
                          snapshot.data?.docs.isEmpty == true) {
                        return Center(
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(
                                Icons.chat_bubble_outline,
                                size: 48,
                                color: Colors.grey,
                              ),
                              SizedBox(height: 16),
                              Text(
                                'No saved history yet',
                                style: GoogleFonts.inter(color: Colors.grey),
                              ),
                              SizedBox(height: 8),
                              Text(
                                'Current session: ${_messages.length} messages',
                                style: GoogleFonts.inter(
                                  color: Colors.grey,
                                  fontSize: 12,
                                ),
                              ),
                            ],
                          ),
                        );
                      }

                      var docs = (snapshot.data as QuerySnapshot).docs;
                      return Column(
                        children: [
                          // Add button to send history to endpoint
                          Padding(
                            padding: EdgeInsets.all(16),
                            child: ElevatedButton.icon(
                              onPressed: () => _sendHistoryToEndpoint(docs),
                              icon: Icon(Icons.cloud_upload),
                              label: Text('Send History to AI Model'),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Colors.blue,
                                foregroundColor: Colors.white,
                              ),
                            ),
                          ),
                          Expanded(
                            child: ListView.builder(
                              itemCount: docs.length,
                              itemBuilder: (context, index) {
                                var message = docs[index];
                                var data =
                                    message.data() as Map<String, dynamic>;

                                return Container(
                                  margin: EdgeInsets.only(
                                    bottom: 8,
                                    left: 16,
                                    right: 16,
                                  ),
                                  padding: EdgeInsets.all(12),
                                  decoration: BoxDecoration(
                                    color:
                                        data['isUser'] == true
                                            ? Colors.blue[50]
                                            : Colors.grey[50],
                                    borderRadius: BorderRadius.circular(8),
                                  ),
                                  child: Column(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.start,
                                    children: [
                                      Text(
                                        data['isUser'] == true
                                            ? 'You'
                                            : 'AI Assistant',
                                        style: GoogleFonts.inter(
                                          fontWeight: FontWeight.bold,
                                          color:
                                              data['isUser'] == true
                                                  ? Colors.blue
                                                  : Colors.green,
                                        ),
                                      ),
                                      SizedBox(height: 4),
                                      Text(
                                        data['message'] ?? '',
                                        style: GoogleFonts.inter(),
                                      ),
                                      if (data['timestamp'] != null)
                                        Padding(
                                          padding: EdgeInsets.only(top: 4),
                                          child: Text(
                                            data['timestamp']
                                                .toDate()
                                                .toString()
                                                .split('.')[0],
                                            style: GoogleFonts.inter(
                                              fontSize: 12,
                                              color: Colors.grey,
                                            ),
                                          ),
                                        ),
                                    ],
                                  ),
                                );
                              },
                            ),
                          ),
                        ],
                      );
                    },
                  ),
                ),
                SizedBox(height: 16),
                ElevatedButton(
                  onPressed: () {
                    _firebaseService.clearChatHistory();
                    Navigator.pop(context);
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(content: Text('Chat history cleared')),
                    );
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.red,
                    foregroundColor: Colors.white,
                  ),
                  child: Text('Clear History'),
                ),
              ],
            ),
          ),
    );
  }

  void _sendMessage() async {
    final text = _textController.text.trim();

    if (text.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Please enter a message'),
          backgroundColor: Colors.orange,
        ),
      );
      return;
    }

    // Store current image reference BEFORE clearing it
    final imageToSend = _selectedImage;
    final messageText = text.isEmpty ? 'Image uploaded' : text;

    // Add user message
    setState(() {
      _messages.add(
        ChatMessage(
          text: messageText,
          image: _selectedImage,
          isUser: true,
          timestamp: DateTime.now(),
        ),
      );

      // Clear input
      _textController.clear();
      _selectedImage = null; // Clear AFTER storing the reference
    });

    _scrollToBottom();

    // Show loading indicator
    setState(() {
      _messages.add(
        ChatMessage(
          text: 'Analyzing...',
          image: null,
          isUser: false,
          timestamp: DateTime.now(),
        ),
      );
    });

    _scrollToBottom();

    try {
      // Save user message to Firebase
      await _firebaseService.saveChatMessage(
        message: text.isEmpty ? 'Please analyze this plant image' : text,
        isUser: true,
        imagePath: imageToSend?.path,
      );

      // Call API with stored message and image
      String response = await ApiService.diagnosePlant(
        message:
            messageText == 'Image uploaded'
                ? 'Please analyze this plant image'
                : messageText,
        image: imageToSend,
      );

      // Save AI response to Firebase
      await _firebaseService.saveChatMessage(message: response, isUser: false);

      // Save complete diagnosis to Firebase
      await _firebaseService.saveDiagnosisResult(
        userMessage: text.isEmpty ? 'Please analyze this plant image' : text,
        aiResponse: response,
        imagePath: imageToSend?.path,
      );

      // Automatically send chat history to endpoint after each interaction
      try {
        final chatHistorySnapshot =
            await _firebaseService.getChatHistory().first;
        if (chatHistorySnapshot.docs.isNotEmpty) {
          await _sendHistoryToEndpoint(chatHistorySnapshot.docs);
          print('✅ Chat history automatically sent to endpoint');
        }
      } catch (e) {
        print('⚠️ Failed to send chat history: $e');
      }

      // Remove loading message and add actual response
      setState(() {
        _messages.removeLast(); // Remove "Analyzing..." message
        _messages.add(
          ChatMessage(
            text: response,
            image: null,
            isUser: false,
            timestamp: DateTime.now(),
          ),
        );
      });
    } catch (e) {
      // Remove loading message and add error message
      setState(() {
        _messages.removeLast(); // Remove "Analyzing..." message
        _messages.add(
          ChatMessage(
            text:
                'Sorry, I couldn\'t process your request. Please try again. Error: $e',
            image: null,
            isUser: false,
            timestamp: DateTime.now(),
          ),
        );
      });
    }

    _scrollToBottom();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          image: DecorationImage(
            image: AssetImage('assets/bg.png'),
            fit: BoxFit.cover,
          ),
        ),
        child: SafeArea(
          child: FadeTransition(
            opacity: _fadeAnimation,
            child: SlideTransition(
              position: _slideAnimation,
              child: Column(
                children: [
                  // Header with back button and logo
                  Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Row(
                      children: [
                        GestureDetector(
                          onTap: () => Navigator.pop(context),
                          child: Container(
                            padding: const EdgeInsets.all(8),
                            decoration: BoxDecoration(
                              color: Color(0xA0F0F0F0),
                              borderRadius: BorderRadius.circular(12),
                            ),
                            child: Icon(
                              Icons.arrow_back,
                              color: Colors.black87,
                              size: 24,
                            ),
                          ),
                        ),
                        const SizedBox(width: 16),
                        Expanded(
                          child: Center(
                            child: Image.asset('assets/logo.png', height: 60),
                          ),
                        ),
                        const SizedBox(width: 16),
                        GestureDetector(
                          onTap: _showChatHistory,
                          child: Container(
                            padding: const EdgeInsets.all(8),
                            decoration: BoxDecoration(
                              color: Color(0xA0F0F0F0),
                              borderRadius: BorderRadius.circular(12),
                            ),
                            child: Icon(
                              Icons.history,
                              color: Colors.black87,
                              size: 24,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),

                  // Chat messages
                  Expanded(
                    child:
                        _messages.isEmpty
                            ? Center(
                              child: Text(
                                'Start your conversation!',
                                style: GoogleFonts.inter(
                                  fontSize: 18,
                                  color: Colors.black54,
                                  fontWeight: FontWeight.w500,
                                ),
                              ),
                            )
                            : ListView.builder(
                              controller: _scrollController,
                              padding: const EdgeInsets.symmetric(
                                horizontal: 16,
                              ),
                              itemCount: _messages.length,
                              itemBuilder: (context, index) {
                                final message = _messages[index];
                                return _buildMessageBubble(message);
                              },
                            ),
                  ),

                  // Input area
                  _buildInputArea(),
                  const SizedBox(height: 20),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildMessageBubble(ChatMessage message) {
    return Align(
      alignment: message.isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 6, horizontal: 4),
        padding: const EdgeInsets.all(12),
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.75,
        ),
        decoration: BoxDecoration(
          color:
              message.isUser
                  ? Color(0xA0F0F0F0)
                  : Colors.white.withOpacity(0.9),
          borderRadius: BorderRadius.circular(18),
          border:
              message.isUser
                  ? Border.all(color: Colors.green.withOpacity(0.3))
                  : null,
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.1),
              blurRadius: 4,
              offset: Offset(0, 2),
            ),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (message.image != null) ...[
              ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: Image.file(
                  message.image!,
                  height: 200,
                  width: double.infinity,
                  fit: BoxFit.cover,
                ),
              ),
              const SizedBox(height: 8),
            ],
            Text(
              message.text,
              style: GoogleFonts.inter(
                fontSize: 15,
                color: Colors.black87,
                height: 1.4,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              '${message.timestamp.hour.toString().padLeft(2, '0')}:${message.timestamp.minute.toString().padLeft(2, '0')}',
              style: GoogleFonts.inter(fontSize: 12, color: Colors.black54),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildInputArea() {
    return Column(
      children: [
        // Display selected image name if any
        if (_selectedImage != null)
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 5),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 15, vertical: 8),
              decoration: BoxDecoration(
                color: Color(0xA0F0F0F0),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                  color: Colors.green.withOpacity(0.3),
                  width: 1,
                ),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.image, size: 16, color: Colors.green),
                  const SizedBox(width: 8),
                  Flexible(
                    child: Text(
                      _selectedImage!.path.split('/').last,
                      style: GoogleFonts.inter(
                        fontSize: 12,
                        color: Colors.black87,
                        fontWeight: FontWeight.w500,
                      ),
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  const SizedBox(width: 8),
                  GestureDetector(
                    onTap: () => setState(() => _selectedImage = null),
                    child: Icon(Icons.close, size: 16, color: Colors.red),
                  ),
                ],
              ),
            ),
          ),

        // Input row
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20),
          child: Row(
            children: [
              // Text field container
              Expanded(
                child: Container(
                  decoration: ShapeDecoration(
                    color: Color(0xA0F0F0F0),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(30),
                    ),
                  ),
                  child: Row(
                    children: [
                      // Plus button inside text field on the left
                      GestureDetector(
                        onTap: _showImagePickerDialog,
                        child: Container(
                          margin: const EdgeInsets.all(8),
                          width: 44,
                          height: 44,
                          decoration: ShapeDecoration(
                            color: Color(0xA0F0F0F0),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(22),
                            ),
                          ),
                          child: Icon(
                            Icons.add,
                            color: Colors.black87,
                            size: 24,
                          ),
                        ),
                      ),
                      Expanded(
                        child: Padding(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 12,
                            vertical: 12,
                          ),
                          child: TextField(
                            controller: _textController,
                            maxLines: null,
                            decoration: InputDecoration(
                              hintText: 'Type your message...',
                              hintStyle: GoogleFonts.inter(
                                fontSize: 16,
                                color: Colors.black54,
                              ),
                              border: InputBorder.none,
                            ),
                            style: GoogleFonts.inter(
                              fontSize: 16,
                              color: Colors.black87,
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(width: 12),
              // Send button with liquid glass effect
              GestureDetector(
                onTap: _sendMessage,
                child: Container(
                  width: 56,
                  height: 56,
                  child: Stack(
                    children: [
                      // Shadow layer (the subtle shadow effect)
                      Positioned(
                        left: 0,
                        top: 0,
                        child: Container(
                          width: 56,
                          height: 56,
                          decoration: ShapeDecoration(
                            color: Colors.black.withOpacity(0.10),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(28),
                            ),
                          ),
                        ),
                      ),
                      // Main glass layer (the visible glass effect)
                      Positioned(
                        left: 0,
                        top: 0,
                        child: Container(
                          width: 56,
                          height: 56,
                          decoration: ShapeDecoration(
                            color: Color(0xA0F0F0F0),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(28),
                            ),
                          ),
                        ),
                      ),
                      // Send icon in center
                      Positioned(
                        left: 0,
                        top: 0,
                        child: Container(
                          width: 56,
                          height: 56,
                          child: Icon(
                            Icons.arrow_upward,
                            color: Colors.green.withOpacity(0.8),
                            size: 24,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}
