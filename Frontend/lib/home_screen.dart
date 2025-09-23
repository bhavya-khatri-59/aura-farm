import 'dart:io';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'chat_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  File? _selectedImage;
  final ImagePicker _picker = ImagePicker();
  final TextEditingController _textController = TextEditingController();

  @override
  void dispose() {
    _textController.dispose();
    super.dispose();
  }

  void _navigateToChat() {
    final text = _textController.text.trim();

    if (text.isEmpty && _selectedImage == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Please enter a message or select an image'),
          backgroundColor: Colors.orange,
        ),
      );
      return;
    }

    // Show sending feedback
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Row(
          children: [
            SizedBox(
              width: 16,
              height: 16,
              child: CircularProgressIndicator(
                strokeWidth: 2,
                valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
              ),
            ),
            SizedBox(width: 12),
            Text('Sending to chat...'),
          ],
        ),
        backgroundColor: Colors.green,
        duration: Duration(seconds: 1),
      ),
    );

    // Navigate to chat screen with initial data
    Navigator.push(
      context,
      PageRouteBuilder(
        pageBuilder:
            (context, animation, secondaryAnimation) => ChatScreen(
              initialMessage: text.isNotEmpty ? text : null,
              initialImage: _selectedImage,
            ),
        transitionsBuilder: (context, animation, secondaryAnimation, child) {
          const begin = Offset(1.0, 0.0);
          const end = Offset.zero;
          const curve = Curves.easeInOutCubic;

          var tween = Tween(
            begin: begin,
            end: end,
          ).chain(CurveTween(curve: curve));

          return SlideTransition(
            position: animation.drive(tween),
            child: child,
          );
        },
        transitionDuration: Duration(milliseconds: 600),
      ),
    ).then((_) {
      // Clear the input when returning from chat
      setState(() {
        _textController.clear();
        _selectedImage = null;
      });
    });
  }

  void _navigateToChatDirectly() {
    // Navigate to chat screen without any initial data
    Navigator.push(
      context,
      PageRouteBuilder(
        pageBuilder: (context, animation, secondaryAnimation) => ChatScreen(),
        transitionsBuilder: (context, animation, secondaryAnimation, child) {
          const begin = Offset(1.0, 0.0);
          const end = Offset.zero;
          const curve = Curves.easeInOutCubic;

          var tween = Tween(
            begin: begin,
            end: end,
          ).chain(CurveTween(curve: curve));

          return SlideTransition(
            position: animation.drive(tween),
            child: child,
          );
        },
        transitionDuration: Duration(milliseconds: 600),
      ),
    );
  }

  // Function to handle Plant Health tap
  void _handlePlantHealthTap() {
    // Set the text in the text field
    _textController.text = 'Analyse the plant picture for me';
    // Open the image picker with auto-navigation to chat
    _showImagePickerDialogWithNavigation();
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

  Future<void> _showImagePickerDialogWithNavigation() async {
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
                  _pickImageAndNavigate(ImageSource.gallery);
                },
              ),
              ListTile(
                leading: Icon(Icons.photo_camera),
                title: Text('Take a Photo'),
                onTap: () {
                  Navigator.pop(context);
                  _pickImageAndNavigate(ImageSource.camera);
                },
              ),
            ],
          ),
        );
      },
    );
  }

  Future<void> _pickImageAndNavigate(ImageSource source) async {
    try {
      // Show loading indicator
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Row(
            children: [
              SizedBox(
                width: 20,
                height: 20,
                child: CircularProgressIndicator(
                  strokeWidth: 2,
                  valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                ),
              ),
              SizedBox(width: 10),
              Text(
                'Opening ${source == ImageSource.camera ? 'camera' : 'gallery'}...',
              ),
            ],
          ),
          duration: Duration(seconds: 2),
        ),
      );

      final XFile? pickedFile = await _picker.pickImage(
        source: source,
        maxWidth: 1800,
        maxHeight: 1800,
        imageQuality: 80,
        preferredCameraDevice: CameraDevice.rear,
      );

      // Clear loading message
      ScaffoldMessenger.of(context).hideCurrentSnackBar();

      if (pickedFile != null) {
        setState(() {
          _selectedImage = File(pickedFile.path);
        });

        // Show success message briefly
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Row(
              children: [
                Icon(Icons.check_circle, color: Colors.white),
                SizedBox(width: 10),
                Text('Image selected! Navigating to chat...'),
              ],
            ),
            backgroundColor: Colors.green,
            duration: Duration(seconds: 1),
          ),
        );

        // Wait a moment for the success message, then navigate
        await Future.delayed(Duration(milliseconds: 1200));
        _navigateToChat();
      } else {
        // User cancelled selection
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('No image selected'),
            backgroundColor: Colors.orange,
          ),
        );
      }
    } catch (e) {
      // Clear loading message
      ScaffoldMessenger.of(context).hideCurrentSnackBar();

      // Show error message
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error picking image: $e'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      // Show loading indicator
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Row(
            children: [
              SizedBox(
                width: 20,
                height: 20,
                child: CircularProgressIndicator(
                  strokeWidth: 2,
                  valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                ),
              ),
              SizedBox(width: 10),
              Text(
                'Opening ${source == ImageSource.camera ? 'camera' : 'gallery'}...',
              ),
            ],
          ),
          duration: Duration(seconds: 2),
        ),
      );

      final XFile? pickedFile = await _picker.pickImage(
        source: source,
        maxWidth: 1800,
        maxHeight: 1800,
        imageQuality: 80,
        preferredCameraDevice: CameraDevice.rear,
      );

      // Clear loading message
      ScaffoldMessenger.of(context).hideCurrentSnackBar();

      if (pickedFile != null) {
        setState(() {
          _selectedImage = File(pickedFile.path);
        });

        // Show success message
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Row(
              children: [
                Icon(Icons.check_circle, color: Colors.white),
                SizedBox(width: 10),
                Text('Image selected successfully!'),
              ],
            ),
            backgroundColor: Colors.green,
            duration: Duration(seconds: 2),
          ),
        );
      } else {
        // User cancelled selection
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Image selection cancelled'),
            backgroundColor: Colors.orange,
            duration: Duration(seconds: 1),
          ),
        );
      }
    } catch (e) {
      // Clear loading message
      ScaffoldMessenger.of(context).hideCurrentSnackBar();

      String errorMessage;
      if (e.toString().contains('platform_exception')) {
        errorMessage =
            'Unable to access ${source == ImageSource.camera ? 'camera' : 'gallery'}. Please check app permissions.';
      } else if (e.toString().contains('connection')) {
        errorMessage =
            'Connection error. Try restarting the app or use a physical device.';
      } else {
        errorMessage = 'Error: ${e.toString()}';
      }

      // Show error message with action button
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Icon(Icons.error, color: Colors.white),
                  SizedBox(width: 10),
                  Expanded(child: Text('Failed to select image')),
                ],
              ),
              SizedBox(height: 5),
              Text(
                errorMessage,
                style: TextStyle(fontSize: 12, color: Colors.white70),
              ),
            ],
          ),
          backgroundColor: Colors.red,
          duration: Duration(seconds: 4),
          action: SnackBarAction(
            label: 'Retry',
            textColor: Colors.white,
            onPressed: () => _pickImage(source),
          ),
        ),
      );
    }
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
          child: Column(
            children: [
              // Top section with logo and chat button
              Stack(
                children: [
                  // Logo in center
                  Padding(
                    padding: const EdgeInsets.only(top: 1.0),
                    child: Center(
                      child: Image.asset('assets/logo.png', height: 120),
                    ),
                  ),
                  // Chat button in top-right
                  Positioned(
                    top: 20,
                    right: 20,
                    child: GestureDetector(
                      onTap: _navigateToChatDirectly,
                      child: Container(
                        width: 50,
                        height: 50,
                        decoration: ShapeDecoration(
                          color: Color(0xA0F0F0F0),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(25),
                          ),
                        ),
                        child: Icon(
                          Icons.chat_bubble_outline,
                          color: Colors.black87,
                          size: 24,
                        ),
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 20),
              Expanded(
                child: Column(
                  children: [
                    RichText(
                      textAlign: TextAlign.center,
                      text: TextSpan(
                        children: [
                          TextSpan(
                            text: 'Hi there, ',
                            style: GoogleFonts.roboto(
                              fontSize: 32,
                              fontWeight: FontWeight.bold,
                              color: Colors.black,
                            ),
                          ),
                          WidgetSpan(
                            alignment: PlaceholderAlignment.baseline,
                            baseline: TextBaseline.alphabetic,
                            child: ShaderMask(
                              shaderCallback:
                                  (bounds) => const LinearGradient(
                                    colors: [
                                      Color(0xFF00FF6A),
                                      Color(0xFF4B8AC9),
                                    ],
                                    begin: Alignment.centerLeft,
                                    end: Alignment.centerRight,
                                  ).createShader(bounds),
                              child: Text(
                                'Rajkumar',
                                style: GoogleFonts.roboto(
                                  fontSize: 32,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.white,
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'How can I help you',
                      style: GoogleFonts.roboto(
                        fontSize: 24,
                        fontWeight: FontWeight.w500,
                        color: Colors.black,
                      ),
                    ),
                    const SizedBox(height: 30),
                    // 2x2 Grid of liquid glass boxes
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 20),
                      child: Column(
                        children: [
                          Row(
                            children: [
                              Expanded(
                                child: LiquidGlassRegularMedium(
                                  imagePath: 'assets/Vector.png',
                                  heading: 'Plant Health',
                                  paragraph:
                                      'Take a picture of your plant\'s leaves to analyze.',
                                  onTap: _handlePlantHealthTap,
                                  headingFontSize: 17.0,
                                  paragraphFontSize: 14.0,
                                ),
                              ),
                              const SizedBox(width: 10),
                              Expanded(
                                child: LiquidGlassRegularMedium(
                                  imagePath: 'assets/Vector-1.png',
                                  heading: 'Fertilizer Recommendation',
                                  paragraph:
                                      'Provide your crop and soil details to fertilizer tips and application advice',
                                  headingFontSize: 12.0,
                                  paragraphFontSize: 12.0,
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 10),
                          Row(
                            children: [
                              Expanded(
                                child: LiquidGlassRegularMedium(
                                  imagePath: 'assets/Vector-2.png',
                                  heading: 'Market Insights',
                                  paragraph:
                                      'View current market prices and trends for better selling decisions',
                                  headingFontSize: 16.0,
                                  paragraphFontSize: 12,
                                ),
                              ),
                              const SizedBox(width: 10),
                              Expanded(
                                child: LiquidGlassRegularMedium(
                                  imagePath: 'assets/Vector-3.png',
                                  heading: 'Weather & Irrigation',
                                  paragraph:
                                      'Get localized weather updates and watering recommendations',
                                  headingFontSize: 15.0,
                                  paragraphFontSize: 13.0,
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                    // Push the input to bottom
                    const Spacer(),
                    // Display selected image name if any
                    if (_selectedImage != null)
                      Padding(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 20,
                          vertical: 5,
                        ),
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 15,
                            vertical: 8,
                          ),
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
                                  style: GoogleFonts.roboto(
                                    fontSize: 12,
                                    color: Colors.black87,
                                    fontWeight: FontWeight.w500,
                                  ),
                                  overflow: TextOverflow.ellipsis,
                                ),
                              ),
                              const SizedBox(width: 8),
                              GestureDetector(
                                onTap: () {
                                  setState(() {
                                    _selectedImage = null;
                                  });
                                },
                                child: Icon(
                                  Icons.close,
                                  size: 16,
                                  color: Colors.red,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    // Text input box with liquid glass effect
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 20),
                      child: Row(
                        children: [
                          // Text field container with plus button inside on the right
                          Expanded(
                            child: Container(
                              height: 60,
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
                                          borderRadius: BorderRadius.circular(
                                            22,
                                          ),
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
                                      ),
                                      child: TextField(
                                        controller: _textController,
                                        decoration: InputDecoration(
                                          hintText: 'Ask anything',
                                          hintStyle: GoogleFonts.roboto(
                                            fontSize: 18,
                                            color: Colors.black54,
                                            fontWeight: FontWeight.w500,
                                          ),
                                          border: InputBorder.none,
                                        ),
                                        style: GoogleFonts.roboto(
                                          fontSize: 14,
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
                          // Send button
                          GestureDetector(
                            onTap: _navigateToChat,
                            child: Container(
                              width: 60,
                              height: 60,
                              decoration: ShapeDecoration(
                                color: Color(0xA0F0F0F0),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(30),
                                ),
                              ),
                              child: Icon(
                                Icons.arrow_upward,
                                color: Colors.black87,
                                size: 24,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 20),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class LiquidGlassRegularMedium extends StatelessWidget {
  final String imagePath;
  final String heading;
  final String paragraph;
  final VoidCallback? onTap;
  final double? headingFontSize;
  final double? paragraphFontSize;

  const LiquidGlassRegularMedium({
    super.key,
    required this.imagePath,
    required this.heading,
    required this.paragraph,
    this.onTap,
    this.headingFontSize,
    this.paragraphFontSize,
  });

  @override
  Widget build(BuildContext context) {
    // Get screen size for responsive design
    final screenWidth = MediaQuery.of(context).size.width;

    // Calculate responsive dimensions based on available space
    // Since we're using Expanded, we calculate based on half screen width
    final availableWidth = (screenWidth - 60) / 2; // 60 = padding + spacing

    double responsiveSize;
    double fontSize;
    double iconSize;
    double padding;

    if (screenWidth < 350) {
      // Small screens
      responsiveSize = availableWidth.clamp(100.0, 140.0);
      fontSize = 8;
      iconSize = 24;
      padding = 10;
    } else if (screenWidth < 400) {
      // Medium-small screens
      responsiveSize = availableWidth.clamp(120.0, 160.0);
      fontSize = 9;
      iconSize = 28;
      padding = 12;
    } else if (screenWidth < 450) {
      // Medium screens
      responsiveSize = availableWidth.clamp(140.0, 170.0);
      fontSize = 10;
      iconSize = 30;
      padding = 15;
    } else {
      // Large screens
      responsiveSize = availableWidth.clamp(160.0, 180.0);
      fontSize = 11;
      iconSize = 32;
      padding = 18;
    }

    return GestureDetector(
      onTap:
          onTap ??
          () {
            // Default behavior if no onTap is provided
            print('Tapped on box with image: $imagePath');
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text('You tapped the glass box!'),
                duration: Duration(seconds: 1),
              ),
            );
          },
      child: Container(
        width: responsiveSize,
        height: responsiveSize,
        child: Stack(
          children: [
            // Shadow layer (the subtle shadow effect)
            Positioned(
              left: 0,
              top: 0,
              child: Container(
                width: responsiveSize,
                height: responsiveSize,
                decoration: ShapeDecoration(
                  color: Colors.black.withOpacity(0.10),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(24),
                  ),
                ),
              ),
            ),
            // Main glass layer (the visible glass effect)
            Positioned(
              left: 0,
              top: 0,
              child: Container(
                width: responsiveSize,
                height: responsiveSize,
                decoration: ShapeDecoration(
                  color: Color(0xA0F0F0F0),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(24),
                  ),
                ),
              ),
            ),
            // Logo in top left corner
            Positioned(
              left: padding,
              top: padding,
              child: Container(
                width: iconSize,
                height: iconSize,
                child: Image.asset(
                  imagePath,
                  width: iconSize,
                  height: iconSize,
                  fit: BoxFit.contain,
                ),
              ),
            ),
            // Text at bottom left of glass box
            Positioned(
              left: padding - 2,
              right: padding - 2,
              bottom: padding,
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    heading,
                    style: GoogleFonts.roboto(
                      fontSize: headingFontSize ?? (fontSize + 1),
                      fontWeight: FontWeight.w700,
                      color: Colors.black87,
                      height: 1.1,
                    ),
                    textAlign: TextAlign.left,
                    maxLines: 3,
                    overflow: TextOverflow.ellipsis,
                  ),
                  SizedBox(height: 2),
                  Text(
                    paragraph,
                    style: GoogleFonts.roboto(
                      fontSize: paragraphFontSize ?? (fontSize - 1),
                      fontWeight: FontWeight.w400,
                      color: Colors.black54,
                      height: 1.2,
                    ),
                    textAlign: TextAlign.left,
                    maxLines: 4,
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
