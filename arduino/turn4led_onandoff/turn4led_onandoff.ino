//Interval timer object
IntervalTimer ledTimer;

// LED DIGITAL PINOUT MAPPINGS
// PROGRAMMED LEDS
#define DIGITAL1  0
#define DIGITAL2  1
#define DIGITAL3  2
#define DIGITAL4  3

// SIGNALING LEDS
#define DIGITAL5  4
#define DIGITAL6  5
#define DIGITAL7  6
#define DIGITAL8  7

//incoming serial trigger signal from BIAS
char serial_trig = ' ';
int ledState = LOW;
unsigned long prevTime_ms = 0UL;
unsigned long led_on_interval = 400UL; // time in ms
int isProgrammingledsOn = 1;
int isSequence = 1;
int current_pin_id = 3;

//number of signalling pins
const byte numSignaling_leds = 4;
byte output_led_state [] = {DIGITAL5, DIGITAL6, DIGITAL7, DIGITAL8};
byte input_led_state [] = {DIGITAL1, DIGITAL2, DIGITAL3, DIGITAL4};
byte input_led_signal [] = {'w', 'x', 'y', 'z'};


void wait()
{
  unsigned long currTime_ms = millis();
  prevTime_ms = currTime_ms;
  while (currTime_ms - prevTime_ms < led_on_interval)
  {
    currTime_ms = millis();
  }
}

void ledOnAndOff(int pin , int ledState) {

  digitalWrite(pin, ledState);
}

void ledPattern() {

  if (ledState == LOW) {
    ledState = HIGH;
  } else {
    ledState = LOW;
  }

  for (byte led_id = 0; led_id < numSignaling_leds; led_id++) {
    ledOnAndOff(input_led_state[led_id], ledState);
    delay(1000);
  }
}

void singleLedOnAndOff() {

  ledOnAndOff(DIGITAL1, HIGH);
  wait();
  ledOnAndOff(DIGITAL1, LOW);
  wait();
  ledOnAndOff(DIGITAL2, HIGH);
  wait();
  ledOnAndOff(DIGITAL2, LOW);
  wait();
  ledOnAndOff(DIGITAL3, HIGH);
  wait();
  ledOnAndOff(DIGITAL3, LOW);
  wait();
  ledOnAndOff(DIGITAL4, HIGH);
  wait();
  ledOnAndOff(DIGITAL4, LOW);
  wait();
}

void changingLedOnAndOff()
{

  if (ledState == LOW) {
    ledState = HIGH;
  } else {
    ledState = LOW;
  }

  digitalWrite(input_led_state[current_pin_id], ledState);

  if (ledState == LOW) {

    if (current_pin_id == (numSignaling_leds - 1))
    {
      current_pin_id = 0;
    } else {
      current_pin_id = current_pin_id + 1;
    }
  }
}

void allLedsOn() {

  digitalWrite(DIGITAL1, HIGH);
  digitalWrite(DIGITAL2, HIGH);
  digitalWrite(DIGITAL3, HIGH);
  digitalWrite(DIGITAL4, HIGH);
  digitalWrite(DIGITAL5, HIGH);
  digitalWrite(DIGITAL6, HIGH);
  digitalWrite(DIGITAL7, HIGH);
  digitalWrite(DIGITAL8, HIGH);
}

void singleLedPattern() {

  if (ledState == LOW) {
    ledState = HIGH;
  } else {
    ledState = LOW;
  }

  digitalWrite(input_led_state[current_pin_id], ledState);
}

void convertSerialtoBinary(char serial_input) {

  // turn on signalling leds
  byte state;
  for (byte led_id = 0; led_id < numSignaling_leds; led_id++) {
    
    //if (current_pin_id == led_id){
        state = bitRead(serial_input, led_id);
        digitalWrite(output_led_state[led_id], state);
    //}
    
  }
}

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);

  // set LED pins as output
  pinMode(DIGITAL1, OUTPUT);
  pinMode(DIGITAL2, OUTPUT);
  pinMode(DIGITAL3, OUTPUT);
  pinMode(DIGITAL4, OUTPUT);
  pinMode(DIGITAL5, OUTPUT);
  pinMode(DIGITAL6, OUTPUT);
  pinMode(DIGITAL7, OUTPUT);
  pinMode(DIGITAL8, OUTPUT);

  //set LED pins to low
  digitalWrite(DIGITAL1, LOW);
  digitalWrite(DIGITAL2, LOW);
  digitalWrite(DIGITAL3, LOW);
  digitalWrite(DIGITAL4, LOW);
  digitalWrite(DIGITAL5, LOW);
  digitalWrite(DIGITAL6, LOW);
  digitalWrite(DIGITAL7, LOW);
  digitalWrite(DIGITAL8, LOW);

  if(isProgrammingledsOn){
    if(isSequence)
      //ledTimer.begin(singleLedOnAndOff, 400000); // turn on sequence every 0.40 secs - train  
      ledTimer.begin(singleLedOnAndOff, 3200000);
    else
      //ledTimer.begin(changingLedOnAndOff,4000000);
      ledTimer.begin(singleLedPattern, 400000);
      //allLedsOn();
  }

}

void loop() {

  //input from usb serial port direct from BIAS computer
  if (Serial.available() > 0) {
    
    serial_trig = Serial.read();
    convertSerialtoBinary(serial_trig);   

  }

}
