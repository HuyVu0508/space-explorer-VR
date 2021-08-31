using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class following : MonoBehaviour
{
    public GameObject player;
    public float speed;
    public float keep_distance;  


    // Start is called before the first frame update
    void Start()
    {
    }

    // Update is called once per frame
    void Update()
    {
        Vector3 current_position = transform.position;
        Vector3 distance_vector = player.transform.position - current_position;
        float distance_value = distance_vector.magnitude;
        Vector3 direction = distance_vector/distance_value;

        // Check if object is not close enough to player, move closer to player
        if (distance_value>keep_distance){
            float current_x = transform.position.x;
            float current_y = transform.position.y;
            float current_z = transform.position.z;
            transform.position = new Vector3(current_x + speed * direction.x * Time.deltaTime, current_y, current_z + speed * direction.z * Time.deltaTime);
        }

        // // Floating effect
        // temporary_position.y += Mathf.Sin (Time.fixedTime * Mathf.PI * frequency) * amplitude;
        // transform.position = temporary_position;        
    }

}